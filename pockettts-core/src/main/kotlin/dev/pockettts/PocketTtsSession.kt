package dev.pockettts

import java.io.Closeable
import java.util.Random
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * One utterance's host state (KV cache, position, RNG, per-call counters) over
 * a shared [PocketTtsEngine]. Cheap to create; the graphs and buffers belong to
 * the engine.
 *
 * Three entry points, one per consumer:
 *  * [synthesize] — the whole utterance at once.
 *  * [stream] — blocking, but hands each decoded chunk to a callback as it is
 *    produced (the TTS service and the demo app).
 *  * [speak] / [begin] — asynchronous, cancellable; [begin] takes text
 *    fragments as an agent's LLM emits them and speaks complete sentences
 *    immediately (barge-in via [Utterance.cancel]).
 */
class PocketTtsSession internal constructor(
    private val engine: PocketTtsEngine,
    val voice: String,
) : Closeable {

    /**
     * Playback tempo, 1.0 = natural. Applied as a host-side constant-pitch
     * time-stretch on the decoded PCM ([SonicStretcher]), so the graphs and the
     * RNG stream are unaffected: the same [noiseSeed] gives the same utterance
     * at any rate. A rate below 1 makes the audio longer, like slower speech.
     *
     * Captured when a call starts, so changing it mid-utterance takes effect on
     * the next call.
     */
    @Volatile
    var rate: Float = 1f
        set(value) {
            field = value.coerceIn(SonicStretcher.MIN_SPEED, SonicStretcher.MAX_SPEED)
        }

    /**
     * Voice pitch, 1.0 = natural, applied alongside [rate]. Because the stretch
     * is duration-preserving, changing it alone shifts pitch without changing
     * how long the utterance takes. Captured at call start, like [rate].
     */
    @Volatile
    var pitch: Float = 1f
        set(value) {
            field = value.coerceIn(SonicStretcher.MIN_PITCH, SonicStretcher.MAX_PITCH)
        }

    /** Voices this engine can speak; the engine holds the graphs, the set the files. */
    val voices: List<Voice> get() = engine.voices

    private val G = PocketTts.G
    private val PMAX = PocketTts.PMAX
    private val HD = PocketTts.HD
    private val NH = PocketTts.NH
    private val LDIM = PocketTts.LDIM

    // ---- utterance state --------------------------------------------------
    // K/V, mask and decoder scratch are the engine's; access is serialized, so
    // they are reset (not re-allocated) per utterance.
    private val pk = engine.pk
    private val pv = engine.pv
    private val mask = engine.mask
    private var pos = 0

    private var voiceName = ""
    private var voiceK = FloatArray(0)
    private var voiceV = FloatArray(0)
    private var voiceLen = 0

    private val cosArr = FloatArray(HD)
    private val sinArr = FloatArray(HD)
    private val invFreq = DoubleArray(HD / 2) { 1.0 / Math.pow(PocketTts.THETA, it / 32.0) }
    private val rnd = Random(engine.noiseSeed ?: System.nanoTime())
    private val zeroNoise = FloatArray(LDIM)

    private val steps = engine.lmSteps
    private val msCos = FloatArray(steps * HD)
    private val msSin = FloatArray(steps * HD)
    private val msMask = FloatArray(steps * (PMAX + 1))
    private val msWrite = FloatArray(steps * PMAX)
    private val msNoise = FloatArray(steps * LDIM)

    @Volatile
    private var cancelled = false

    // ---- per-call accumulators -------------------------------------------
    private var sLmIn = 0L
    private var sLmRun = 0L
    private var sLmRead = 0L
    private var sLmSteps = 0
    private var sLmInv = 0
    private var sLmInBytes = 0L
    private var sLmOutBytes = 0L
    private var sPrompt = 0
    private var sFrames = 0
    private var sDecTx = 0L
    private var sSeanet = 0L
    private var sChunks = 0
    private var sFirstChunk = -1L
    private var sAudioChunks = 0

    /** Bind a repacked voice state; the engine caches the decoded arrays. */
    fun loadVoice(name: String) {
        if (name == voiceName) return
        val state = engine.voiceState(name)
        voiceK = state.k
        voiceV = state.v
        voiceLen = state.len
        voiceName = name
    }

    private fun resetToVoice() {
        pk.fill(0f); pv.fill(0f)
        for (g in 0 until G) {
            System.arraycopy(voiceK, g * voiceLen * HD, pk, g * PMAX * HD, voiceLen * HD)
            System.arraycopy(voiceV, g * voiceLen * HD, pv, g * PMAX * HD, voiceLen * HD)
        }
        mask.fill(PocketTts.MASK_NEG)
        for (h in 0 until NH) {
            val base = h * (PMAX + 1)
            for (p in 0 until voiceLen) mask[base + p] = 0f
            mask[base + PMAX] = 0f
        }
        pos = voiceLen
    }

    private fun embRow(id: Int): FloatArray {
        val out = FloatArray(PocketTts.H)
        var b = id * PocketTts.H * 2
        for (j in 0 until PocketTts.H) {
            out[j] = android.util.Half.toFloat(engine.embMap.getShort(b)); b += 2
        }
        return out
    }

    private fun projectLatent(lat: FloatArray): FloatArray {
        val out = FloatArray(PocketTts.H)
        for (o in 0 until PocketTts.H) {
            var acc = 0f
            val row = o * LDIM
            for (i in 0 until LDIM) acc += engine.inputLinear[row + i] * lat[i]
            out[o] = acc
        }
        return out
    }

    private fun ropeFill(p: Int) {
        for (j in 0 until HD / 2) {
            val ang = p * invFreq[j]
            val c = cos(ang).toFloat(); val s = sin(ang).toFloat()
            cosArr[j] = c; cosArr[j + HD / 2] = c
            sinArr[j] = s; sinArr[j + HD / 2] = s
        }
    }

    private fun gaussNoise(): FloatArray = FloatArray(LDIM) {
        (rnd.nextGaussian() * sqrt(PocketTts.TEMP.toDouble())).toFloat()
    }

    /**
     * Append the prompt tokens to the packed KV. With the fused graph's
     * head-less `prefill` signature this is one invocation per
     * [PocketTts.PREFILL_TOKENS] tokens instead of one fused step (flow head
     * included) per token -- the latent a prompt step computes is discarded --
     * and the host never uploads the ~25 MB packed KV per token. Falls back to
     * the per-token fused step on model drops that predate the signature.
     */
    private fun prefill(ids: IntArray) {
        val ins = engine.prefillIn
        val outs = engine.prefillOut
        if (ins == null || outs == null) {
            for (id in ids) step(embRow(id), zeroNoise)
            return
        }
        check(pos + ids.size <= PMAX) { "KV cache overflow at $pos + ${ids.size}" }
        val p = PocketTts.PREFILL_TOKENS
        val emb = engine.prefillEmb
        val cos = engine.prefillCos
        val sin = engine.prefillSin
        val msk = engine.prefillMask
        val wr = engine.prefillWrite
        val t0 = System.nanoTime()
        var runNs = 0L
        var i = 0
        while (i < ids.size && !cancelled) {
            val n = minOf(p, ids.size - i)
            java.util.Arrays.fill(emb, 0f)
            java.util.Arrays.fill(msk, PocketTts.MASK_NEG)
            java.util.Arrays.fill(wr, 0f)
            for (j in 0 until n) {
                var b = ids[i + j] * PocketTts.H * 2
                val eb = j * PocketTts.H
                for (h in 0 until PocketTts.H) {
                    emb[eb + h] = android.util.Half.toFloat(engine.embMap.getShort(b)); b += 2
                }
                ropeFill(pos + j)
                System.arraycopy(cosArr, 0, cos, j * HD, HD)
                System.arraycopy(sinArr, 0, sin, j * HD, HD)
                val mb = j * (PMAX + 1)
                for (q in 0 until pos + j) msk[mb + q] = 0f
                msk[mb + PMAX] = 0f
                wr[j * PMAX + pos + j] = 1f
            }
            ins[0].writeFloat(emb)
            ins[1].writeFloat(cos)
            ins[2].writeFloat(sin)
            ins[3].writeFloat(msk)
            ins[4].writeFloat(wr)
            ins[5].writeFloat(pk)
            ins[6].writeFloat(pv)
            val rt = System.nanoTime()
            engine.lm.run(ins, outs, 0)
            runNs += System.nanoTime() - rt
            val out = outs[0].readFloat()
            var o = 0
            for (j in 0 until n) {
                for (g in 0 until G) {
                    System.arraycopy(out, o, pk, g * PMAX * HD + (pos + j) * HD, HD); o += HD
                }
            }
            for (j in 0 until n) {
                for (g in 0 until G) {
                    System.arraycopy(out, o, pv, g * PMAX * HD + (pos + j) * HD, HD); o += HD
                }
            }
            for (j in 0 until n) {
                for (h in 0 until NH) mask[h * (PMAX + 1) + pos + j] = 0f
            }
            pos += n
            i += n
        }
        val t1 = System.nanoTime()
        sLmIn += (t1 - t0) - runNs
        sLmRun += runNs
        sLmSteps += ids.size
        sLmInv += (ids.size + p - 1) / p
    }

    /**
     * One fused frame: flow-LM step + flow head in a single invocation.
     * Output layout: eos(1) | latent(32) | new-k(96*64) | new-v(96*64).
     */
    private fun step(emb: FloatArray, noise: FloatArray): Pair<FloatArray, Float> {
        check(pos < PMAX) { "KV cache overflow at $pos" }
        val t0 = System.nanoTime()
        ropeFill(pos)
        engine.lmIn[0].writeFloat(emb)
        engine.lmIn[1].writeFloat(cosArr)
        engine.lmIn[2].writeFloat(sinArr)
        engine.lmIn[3].writeFloat(mask)
        engine.lmIn[4].writeFloat(pk)
        engine.lmIn[5].writeFloat(pv)
        engine.lmIn[6].writeFloat(noise)
        val t1 = System.nanoTime()
        engine.runLm(engine.lmIn, engine.lmOut)
        val t2 = System.nanoTime()
        val out = engine.lmOut[0].readFloat()
        val eos = out[0]
        val latent = out.copyOfRange(1, 1 + LDIM)
        val kvBase = 1 + LDIM
        for (g in 0 until G) {
            System.arraycopy(out, kvBase + g * HD, pk, g * PMAX * HD + pos * HD, HD)
            System.arraycopy(out, kvBase + G * HD + g * HD, pv, g * PMAX * HD + pos * HD, HD)
        }
        for (h in 0 until NH) mask[h * (PMAX + 1) + pos] = 0f
        pos++
        val t3 = System.nanoTime()
        sLmIn += t1 - t0; sLmRun += t2 - t1; sLmRead += t3 - t2; sLmSteps++
        sLmInv++
        sLmInBytes += (
            emb.size + cosArr.size + sinArr.size + mask.size +
                pk.size + pv.size + noise.size
            ).toLong() * Float.SIZE_BYTES
        sLmOutBytes += out.size.toLong() * Float.SIZE_BYTES
        return latent to eos
    }

    /**
     * [steps] frames in one invocation of the multi-step graph. The graph
     * uploads the packed KV once, appends all N new rows internally with the
     * one-hot [msWrite] mask, and returns only the new rows.
     * Output layout: eos[N] | latent[N,32] | new-k[N,G,64] | new-v[N,G,64].
     */
    private fun stepMulti(
        emb: FloatArray,
        noises: Array<FloatArray>,
    ): Pair<Array<FloatArray>, FloatArray> {
        val n = steps
        check(pos + n <= PMAX) { "multi-step KV cache overflow at $pos + $n" }
        val t0 = System.nanoTime()
        java.util.Arrays.fill(msWrite, 0f)
        for (i in 0 until n) {
            ropeFill(pos + i)
            System.arraycopy(cosArr, 0, msCos, i * HD, HD)
            System.arraycopy(sinArr, 0, msSin, i * HD, HD)
            val mb = i * (PMAX + 1)
            for (p in 0..PMAX) msMask[mb + p] = if (p < pos + i || p == PMAX) 0f else PocketTts.MASK_NEG
            msWrite[i * PMAX + (pos + i)] = 1f
            System.arraycopy(noises[i], 0, msNoise, i * LDIM, LDIM)
        }
        val ins = requireNotNull(engine.lmMsIn) { "multi-step graph not loaded" }
        val outs = requireNotNull(engine.lmMsOut)
        val model = requireNotNull(engine.lmMs)
        ins[0].writeFloat(emb)
        ins[1].writeFloat(msCos)
        ins[2].writeFloat(msSin)
        ins[3].writeFloat(msMask)
        ins[4].writeFloat(msWrite)
        ins[5].writeFloat(pk)
        ins[6].writeFloat(pv)
        ins[7].writeFloat(msNoise)
        val t1 = System.nanoTime()
        model.run(ins, outs)
        val t2 = System.nanoTime()
        val out = outs[0].readFloat()
        val eos = FloatArray(n)
        val lats = Array(n) { FloatArray(LDIM) }
        for (i in 0 until n) {
            eos[i] = out[i]
            System.arraycopy(out, n + i * LDIM, lats[i], 0, LDIM)
        }
        var o = n * (1 + LDIM)
        for (i in 0 until n) {
            val p = pos + i
            for (g in 0 until G) {
                System.arraycopy(out, o, pk, g * PMAX * HD + p * HD, HD); o += HD
            }
        }
        for (i in 0 until n) {
            val p = pos + i
            for (g in 0 until G) {
                System.arraycopy(out, o, pv, g * PMAX * HD + p * HD, HD); o += HD
            }
        }
        for (h in 0 until NH) {
            val base = h * (PMAX + 1)
            for (i in 0 until n) mask[base + pos + i] = 0f
        }
        pos += n
        val t3 = System.nanoTime()
        sLmIn += t1 - t0; sLmRun += t2 - t1; sLmRead += t3 - t2; sLmSteps += n; sLmInv++
        sLmInBytes += (
            PocketTts.H + 2 * n * HD + n * (PMAX + 1) + n * PMAX +
                2 * G * PMAX * HD + n * LDIM
            ).toLong() * Float.SIZE_BYTES
        sLmOutBytes += out.size.toLong() * Float.SIZE_BYTES
        return lats to eos
    }

    // ---- public entry points ---------------------------------------------

    /** Generate the whole utterance, blocking. */
    fun synthesize(text: String): TtsResult = synchronized(engine.lock) {
        val t0 = System.nanoTime()
        engine.noiseSeed?.let { rnd.setSeed(it) }
        resetProfile()
        loadVoice(voice)
        val audio = ArrayList<FloatArray>()
        var frames = 0
        for (chunk in splitIntoBestSentences(text)) {
            val (prepared, eosGuess) = prepareTextPrompt(chunk)
            val ids = engine.tokenizer.encode(prepared)
            val latents = generateChunk(ids, framesAfterEos = eosGuess + 2)
            android.util.Log.i("PocketTTS", "chunk: ${ids.size} tokens -> ${latents.size} frames")
            frames += latents.size
            sPrompt += ids.size; sFrames += latents.size; sChunks++
            if (latents.isNotEmpty()) audio.add(decode(latents))
        }
        val out = applyShaping(concat(audio))
        TtsResult(out, frames, (System.nanoTime() - t0) / 1_000_000, snapshotProfile())
    }

    /**
     * Streaming synthesis: [onChunk] receives audio as soon as it is decodable.
     * Falls back to [synthesize] (one chunk) when the streaming SEANet graph is
     * absent.
     */
    fun stream(text: String, onChunk: (FloatArray) -> Unit): TtsResult = synchronized(engine.lock) {
        if (engine.deconlyW == null) {
            android.util.Log.w("PocketTTS", "no ${PocketTts.deconlyGraph(engine.streamW)}: one-shot fallback")
            val r = synthesize(text)
            if (r.audio.isNotEmpty()) onChunk(r.audio)
            return@synchronized r
        }
        val t0 = System.nanoTime()
        engine.noiseSeed?.let { rnd.setSeed(it) }
        resetProfile()
        loadVoice(voice)
        val all = ArrayList<FloatArray>()
        var frames = 0
        // Capture both at call start: changing them mid-utterance applies next call.
        val stretcher = SonicStretcher(rate, pitch)
        val textChunks = splitIntoBestSentences(text)
        android.util.Log.i(
            "PocketTTSTime",
            "stream: ${text.length} chars -> ${textChunks.size} text chunk(s), rate=$rate pitch=$pitch",
        )
        for ((si, s) in textChunks.withIndex()) {
            android.util.Log.i(
                "PocketTTSTime",
                "TRACE split #$si/${textChunks.size - 1} chars=${s.length} text=\"${trace(s)}\"",
            )
        }
        var lastAudioAt = 0L
        for ((ci, chunk) in textChunks.withIndex()) {
            val chunkT0 = System.nanoTime()
            val chunkGap = if (lastAudioAt > 0) (chunkT0 - lastAudioAt) / 1_000_000 else -1L
            android.util.Log.i(
                "PocketTTSTime",
                "TRACE chunk #$ci/${textChunks.size - 1} start at " +
                    "${(chunkT0 - t0) / 1_000_000}ms gapSincePrevAudio=${chunkGap}ms " +
                    "text=\"${trace(chunk)}\"",
            )
            val (prepared, eosGuess) = prepareTextPrompt(chunk)
            val ids = engine.tokenizer.encode(prepared)
            var chunkFirstAudio = -1L
            var chunkFirstFrame = -1L
            var chunkAudioChunks = 0
            val dec = StreamDecoder { c ->
                if (sFirstChunk < 0) sFirstChunk = (System.nanoTime() - t0) / 1_000_000
                if (chunkFirstAudio < 0) chunkFirstAudio = (System.nanoTime() - chunkT0) / 1_000_000
                val shaped = stretcher.push(c)
                if (shaped.isNotEmpty()) {
                    all.add(shaped)
                    onChunk(shaped)
                    chunkAudioChunks++
                    lastAudioAt = System.nanoTime()
                }
            }
            val genT = System.nanoTime()
            val lats = generateChunk(ids, framesAfterEos = eosGuess + 2) { lat ->
                if (chunkFirstFrame < 0) chunkFirstFrame = System.nanoTime()
                dec.push(lat)
            }
            val genMs = (System.nanoTime() - genT) / 1_000_000
            val flushT = System.nanoTime()
            dec.flush()
            val flushMs = (System.nanoTime() - flushT) / 1_000_000
            android.util.Log.i(
                "PocketTTSTime",
                "text chunk $ci/${textChunks.size - 1}: ${ids.size} tokens -> ${lats.size} frames " +
                    "gen=${genMs}ms flush=${flushMs}ms firstAudio=${chunkFirstAudio}ms " +
                    "audioChunks=$chunkAudioChunks wall=${
                        (System.nanoTime() - chunkT0) / 1_000_000
                    }ms firstFrame=${if (chunkFirstFrame < 0) -1L else (chunkFirstFrame - chunkT0) / 1_000_000}ms",
            )
            frames += lats.size
            sPrompt += ids.size; sFrames += lats.size; sChunks++
        }
        val tail = stretcher.finish()
        if (tail.isNotEmpty()) {
            all.add(tail)
            onChunk(tail)
        }
        val totalMs = (System.nanoTime() - t0) / 1_000_000
        android.util.Log.i(
            "PocketTTSTime",
            "stream done: ${totalMs}ms frames=$frames audioChunks=${all.size} " +
                "firstAudio=${sFirstChunk}ms lm=${sLmRun / 1_000_000}ms " +
                "decTx=${sDecTx / 1_000_000}ms seanet=${sSeanet / 1_000_000}ms ${stretcher.stats()}",
        )
        TtsResult(concat(all), frames, totalMs, snapshotProfile())
    }

    /**
     * Asynchronous [stream]. The utterance runs on the engine's worker thread;
     * [Utterance.cancel] stops generation at the next frame/chunk boundary.
     */
    fun speak(text: String, listener: SpeechListener): Utterance {
        cancelled = false
        val future = engine.submit {
            try {
                val r = stream(text) { listener.onAudio(it) }
                listener.onDone(r)
            } catch (e: Throwable) {
                listener.onError(e)
            }
        }
        return object : Utterance {
            override fun cancel() {
                this@PocketTtsSession.cancel()
                future.cancel(true)
            }
        }
    }

    /**
     * Low-latency push API for a voice agent: [AgentSpeech.push] text fragments
     * as the LLM emits them; a complete sentence is synthesized as soon as its
     * terminator arrives, so audio starts well before the reply is finished.
     * Generation happens on the calling thread, so call it from your own worker.
     */
    fun begin(listener: SpeechListener): AgentSpeech = Agent(listener)

    /** Set the cancellation flag checked by the generation/decoder loops. */
    fun cancel() {
        cancelled = true
    }

    /** LM-only micro-benchmark: [steps] frames with no Mimi decode. */
    fun microBenchLm(steps: Int): TtsProfile = synchronized(engine.lock) {
        engine.noiseSeed?.let { rnd.setSeed(it) }
        val oldFrames = sFrames
        resetProfile()
        loadVoice(voice)
        resetToVoice()
        val n = minOf(steps, PMAX - pos - 1).coerceAtLeast(0)
        var emb = engine.bosInput
        var g = 0
        while (g < n) {
            if (engine.lmSteps > 1 && g + engine.lmSteps <= n && pos + engine.lmSteps <= PMAX) {
                val (lats, _) = stepMulti(emb, Array(engine.lmSteps) { gaussNoise() })
                emb = projectLatent(lats[engine.lmSteps - 1])
                g += engine.lmSteps
            } else {
                emb = projectLatent(step(emb, gaussNoise()).first)
                g++
            }
        }
        sFrames = n
        val p = snapshotProfile()
        sFrames = oldFrames
        p
    }

    override fun close() {
        // Host state only; the graphs and buffers live on the engine.
        voiceK = FloatArray(0); voiceV = FloatArray(0); voiceName = ""
    }

    // ---- agent implementation --------------------------------------------

    /** Push-style, sentence-at-a-time speech for an agent loop. */
    interface AgentSpeech : Utterance {
        /** Feed text (may end mid-word); complete sentences fire immediately. */
        fun push(text: String)
        /** Flush the remaining text and finish the utterance. */
        fun end()
    }

    private inner class Agent(private val listener: SpeechListener) : AgentSpeech {
        private val pending = StringBuilder()
        private val audio = ArrayList<FloatArray>()
        private var frames = 0
        private var ms = 0L
        private var profile: TtsProfile? = null
        private var ended = false

        override fun push(text: String) {
            if (ended) return
            android.util.Log.i(
                "PocketTTSTime",
                "TRACE agent push chars=${text.length} pending=${pending.length} " +
                    "text=\"${trace(text, 400)}\"",
            )
            pending.append(text)
            while (true) {
                val sentence = takeSentence() ?: break
                speakSentence(sentence)
                if (cancelled) break
            }
        }

        override fun end() {
            if (ended) return
            ended = true
            if (!cancelled) {
                val rest = pending.toString().trim()
                if (rest.isNotEmpty()) speakSentence(rest)
            }
            pending.setLength(0)
            val total = concat(audio)
            listener.onDone(
                TtsResult(
                    total, frames, ms,
                    profile ?: TtsProfile(0, 0, frames, 0, 0, 0, 0, 0, 0, firstChunkMs = -1),
                ),
            )
        }

        override fun cancel() {
            this@PocketTtsSession.cancel()
            pending.setLength(0)
            ended = true
        }

        private fun speakSentence(sentence: String) {
            try {
                val r = stream(sentence) { listener.onAudio(it) }
                audio.add(r.audio)
                frames += r.frames
                ms += r.ms
                profile = r.profile
                android.util.Log.i(
                    "PocketTTSTime",
                    "TRACE agent sentence -> ${r.frames} frames in ${r.ms}ms (total ${ms}ms) " +
                        "text=\"${trace(sentence, 400)}\"",
                )
            } catch (e: Throwable) {
                ended = true
                listener.onError(e)
            }
        }

        /**
         * Pull one complete sentence off [pending]: up to and including a
         * terminator followed by whitespace (or the buffer end), else null.
         */
        private fun takeSentence(): String? {
            for (i in pending.indices) {
                val c = pending[i]
                if (c == '.' || c == '!' || c == '?' || c == '…') {
                    val next = pending.getOrNull(i + 1)
                    if (next == null || next.isWhitespace()) {
                        val s = pending.substring(0, i + 1).trim()
                        pending.delete(0, i + 1)
                        if (s.isNotEmpty()) return s
                        return takeSentence()
                    }
                }
            }
            return null
        }
    }

    // ---- autoregressive loop / decode ------------------------------------

    private fun generateChunk(
        ids: IntArray,
        framesAfterEos: Int,
        sink: ((FloatArray) -> Unit)? = null,
    ): List<FloatArray> {
        resetToVoice()
        val promptT = System.nanoTime()
        prefill(ids)
        val promptMs = (System.nanoTime() - promptT) / 1_000_000
        val estimate = ceil((ids.size / PocketTts.TOKENS_PER_SECOND + PocketTts.GEN_SECONDS_PADDING) * PocketTts.FRAME_RATE)
        val maxGen = minOf(estimate.toInt(), PMAX - pos - 1)
        if (sink != null) {
            android.util.Log.i(
                "PocketTTSTime",
                "TRACE generateChunk prompt ${ids.size} tokens in ${promptMs}ms maxGen=$maxGen",
            )
        }
        val latents = ArrayList<FloatArray>(maxGen)
        var emb = engine.bosInput
        var eosStep = -1
        var g = 0
        while (g < maxGen && !cancelled) {
            if (engine.lmSteps > 1 && g + engine.lmSteps <= maxGen && pos + engine.lmSteps <= PMAX) {
                val (lats, eoses) = stepMulti(emb, Array(engine.lmSteps) { gaussNoise() })
                var stop = false
                for (i in 0 until engine.lmSteps) {
                    if (eoses[i] > PocketTts.EOS_THRESHOLD && eosStep < 0) eosStep = g + i
                    if (eosStep >= 0 && g + i >= eosStep + framesAfterEos) { stop = true; break }
                    latents.add(lats[i]); sink?.invoke(lats[i])
                }
                if (stop || cancelled) break
                emb = projectLatent(lats[engine.lmSteps - 1])
                g += engine.lmSteps
            } else {
                val (lat, eosLogit) = step(emb, gaussNoise())
                if (eosLogit > PocketTts.EOS_THRESHOLD && eosStep < 0) eosStep = g
                if (eosStep >= 0 && g >= eosStep + framesAfterEos) break
                latents.add(lat); sink?.invoke(lat)
                emb = projectLatent(lat)
                g++
            }
        }
        return latents
    }

    /** Mimi decode: overlapped dec_tx blocks -> one-shot SEANet window. */
    private fun decode(latents: List<FloatArray>): FloatArray {
        val tDec = System.nanoTime()
        val t = minOf(latents.size, PocketTts.DEC_FRAMES)
        val feat = engine.decFeat
        val blk = engine.decBlk
        val neutral = engine.neutral

        fun runBlock(prev: FloatArray, start: Int): FloatArray {
            System.arraycopy(prev, 0, blk, 0, LDIM)
            for (f in 0 until PocketTts.F_BLK) {
                val src = if (start + f < t) latents[start + f] else neutral
                System.arraycopy(src, 0, blk, (1 + f) * LDIM, LDIM)
            }
            engine.dectxIn[0].writeFloat(blk)
            engine.dectx.run(engine.dectxIn, engine.dectxOut)
            return engine.dectxOut[0].readFloat()
        }

        var out = runBlock(neutral, 0)
        val n0 = minOf(PocketTts.F_BLK, t)
        for (c in 0 until PocketTts.MIMI_D) {
            System.arraycopy(out, c * PocketTts.S_BLK, feat, c * PocketTts.S_DEC, n0 * PocketTts.UPS)
        }
        var kept = PocketTts.F_BLK
        while (kept < t) {
            val start = kept - PocketTts.F_HOP
            out = runBlock(latents[start - 1], start)
            val n = minOf(PocketTts.F_BLK, t - start)
            val keepN = (n - PocketTts.F_HOP) * PocketTts.UPS
            for (c in 0 until PocketTts.MIMI_D) {
                System.arraycopy(
                    out, c * PocketTts.S_BLK + PocketTts.F_HOP * PocketTts.UPS,
                    feat, c * PocketTts.S_DEC + kept * PocketTts.UPS, keepN,
                )
            }
            kept += n - PocketTts.F_HOP
        }

        val tSeanet = System.nanoTime()
        engine.deconlyIn[0].writeFloat(feat)
        engine.deconly.run(engine.deconlyIn, engine.deconlyOut)
        val wav = engine.deconlyOut[0].readFloat()
        sDecTx += tSeanet - tDec
        sSeanet += System.nanoTime() - tSeanet
        return FloatArray(t * PocketTts.SPF) { wav[it].coerceIn(-1f, 1f) }
    }

    // ---- streaming decode -------------------------------------------------

    /**
     * Incremental Mimi decode. dec_tx blocks run as soon as their inputs exist
     * and SEANet windows as soon as [PocketTts.STREAM_W] positions are ready.
     * Both stages are block-exact, so concatenated chunks equal [decode].
     */
    private inner class StreamDecoder(private val onChunk: (FloatArray) -> Unit) {
        private val lats = ArrayList<FloatArray>()
        private val feat = engine.decFeat
        private val win = engine.streamWin
        private val blk = engine.decBlk
        private var kept = 0
        private var featPos = 0
        private var emitted = 0

        /** Next emission target in frames: cumulative chunk sizes. */
        private var emitAt = PocketTts.STREAM_RAMP[0]
        private var ramp = 1

        /** Largest chunk one SEANet window can add: window minus its left context. */
        private val maxHop = (engine.streamW - PocketTts.STREAM_L) / PocketTts.UPS
        var chunks = 0
            private set

        fun push(lat: FloatArray) {
            if (cancelled) return
            lats.add(lat)
            val n = lats.size
            // Decode up to each emission target as the LM reaches it and emit at
            // once: a partial window is exact because the SEANet is causal, and
            // the growing chunk sizes keep playback fed without stalling on a
            // full window.
            while (!cancelled && kept < n && n >= emitAt) {
                decodeTo(emitAt)
                emitWindow()
                emitAt += nextChunk()
            }
        }

        fun flush() {
            val n = lats.size
            while (kept < n && !cancelled) decodeTo(n)
            while (featPos > emitted && !cancelled) emitWindow()
        }

        /** Next chunk size: the ramp while it lasts, then the steady window. */
        private fun nextChunk(): Int {
            if (ramp >= PocketTts.STREAM_RAMP.size) return maxHop
            return PocketTts.STREAM_RAMP[ramp++]
        }

        /** Run dec_tx blocks until [target] frames are decoded. */
        private fun decodeTo(target: Int) {
            while (kept < target && !cancelled) {
                // Canonical block chain: frame 0 while the prefix is at most
                // F_BLK (whole prefix present, so exact), then slide F_HOP back
                // exactly like the one-shot decode, so the streaming features
                // match it instead of drifting through fp16 boundary rounding.
                val start = if (target <= PocketTts.F_BLK) 0
                else (kept - PocketTts.F_HOP).coerceAtLeast(0)
                val prev = if (start >= 1) lats[start - 1] else engine.neutral
                block(target, start, prev)
            }
        }

        private fun block(size: Int, start: Int, prev: FloatArray) {
            System.arraycopy(prev, 0, blk, 0, LDIM)
            for (f in 0 until PocketTts.F_BLK) {
                val src = if (start + f < size) lats[start + f] else engine.neutral
                System.arraycopy(src, 0, blk, (1 + f) * LDIM, LDIM)
            }
            engine.dectxIn[0].writeFloat(blk)
            val t0 = System.nanoTime()
            engine.dectx.run(engine.dectxIn, engine.dectxOut)
            val out = engine.dectxOut[0].readFloat()
            val ms = (System.nanoTime() - t0) / 1_000_000
            sDecTx += System.nanoTime() - t0
            val drop = if (start == 0) 0 else PocketTts.F_HOP
            val keepN = minOf(PocketTts.F_BLK, size - start) - drop
            android.util.Log.i(
                "PocketTTSTime",
                "  dec_tx block start=$start size=$size drop=$drop keep=$keepN ms=$ms",
            )
            for (c in 0 until PocketTts.MIMI_D) {
                System.arraycopy(
                    out, c * PocketTts.S_BLK + drop * PocketTts.UPS,
                    feat, c * PocketTts.S_DEC + (start + drop) * PocketTts.UPS, keepN * PocketTts.UPS,
                )
            }
            kept = start + drop + keepN
            featPos = kept * PocketTts.UPS
        }

        private fun emitWindow() {
            val model = engine.deconlyW ?: return
            val ins = engine.deconlyWIn ?: return
            val outs = engine.deconlyWOut ?: return
            val w = engine.streamW
            val start = if (emitted == 0) 0 else emitted - PocketTts.STREAM_L
            val avail = minOf(featPos, start + w) - start
            java.util.Arrays.fill(win, 0f)
            for (c in 0 until PocketTts.MIMI_D) {
                System.arraycopy(feat, c * PocketTts.S_DEC + start, win, c * w, avail)
            }
            ins[0].writeFloat(win)
            val t0 = System.nanoTime()
            model.run(ins, outs)
            val wav = outs[0].readFloat()
            val ms = (System.nanoTime() - t0) / 1_000_000
            sSeanet += System.nanoTime() - t0
            val keep = minOf(start + w, featPos) - emitted
            val off = (emitted - start) * PocketTts.SPP
            val out = FloatArray(keep * PocketTts.SPP) { wav[off + it].coerceIn(-1f, 1f) }
            emitted += keep
            chunks++
            sAudioChunks++
            android.util.Log.i(
                "PocketTTSTime",
                "  audio chunk $chunks: ${keep} pos / ${out.size} samples, seanet=${ms}ms, " +
                    "emitted=${emitted / PocketTts.UPS} frames",
            )
            onChunk(out)
        }
    }

    // ---- small helpers ----------------------------------------------------

    /** Post-shape a finished utterance when rate or pitch is not 1.0. */
    private fun applyShaping(audio: FloatArray): FloatArray {
        if (audio.isEmpty()) return audio
        return sonicStretch(audio, rate, pitch)
    }

    private fun concat(parts: List<FloatArray>): FloatArray {
        val total = parts.sumOf { it.size }
        val out = FloatArray(total)
        var o = 0
        for (a in parts) { System.arraycopy(a, 0, out, o, a.size); o += a.size }
        return out
    }

    private fun resetProfile() {
        sLmIn = 0; sLmRun = 0; sLmRead = 0; sLmSteps = 0; sLmInv = 0
        sLmInBytes = 0; sLmOutBytes = 0
        sPrompt = 0; sFrames = 0; sDecTx = 0; sSeanet = 0; sChunks = 0
        sFirstChunk = -1; sAudioChunks = 0
    }

    private fun snapshotProfile() = TtsProfile(
        lmSteps = sLmSteps,
        promptSteps = sPrompt,
        genFrames = sFrames,
        lmInMs = sLmIn / 1_000_000,
        lmRunMs = sLmRun / 1_000_000,
        lmReadMs = sLmRead / 1_000_000,
        decTxMs = sDecTx / 1_000_000,
        seanetMs = sSeanet / 1_000_000,
        chunks = sChunks,
        lmInvocations = sLmInv,
        lmInBytes = sLmInBytes,
        lmOutBytes = sLmOutBytes,
        firstChunkMs = sFirstChunk,
        audioChunks = sAudioChunks,
    )

    // ---- text preparation (ports of pocket_tts.models.tts_model) ----------

    /** Cap/sanitize a text payload so it stays on one readable log line. */
    private fun trace(s: String, max: Int = 1200): String =
        (if (s.length <= max) s else s.substring(0, max) + "…(+" + (s.length - max) + " chars)")
            .replace('\n', ' ')

    internal fun prepareTextPrompt(raw: String): Pair<String, Int> {
        var text = raw.trim()
        require(text.isNotEmpty()) { "Text prompt cannot be empty" }
        text = text.replace('\n', ' ').replace('\r', ' ').replace("  ", " ")
        val words = text.trim().split(Regex("\\s+")).size
        val guess = if (words <= 4) 3 else 1
        if (!text[0].isUpperCase()) text = text[0].uppercaseChar() + text.substring(1)
        if (text.last().isLetterOrDigit()) text += "."
        return text to guess
    }

    internal fun splitIntoBestSentences(raw: String): List<String> {
        val (prepared, _) = prepareTextPrompt(raw)
        val tokens = engine.tokenizer.encode(prepared.trim()).toList()

        fun boundaries(list: List<Int>, marks: Set<Int>): List<Int> {
            val idx = ArrayList<Int>()
            idx.add(0)
            var prevWasBoundary = false
            for ((i, tok) in list.withIndex()) {
                if (tok in marks) prevWasBoundary = true
                else {
                    if (prevWasBoundary) idx.add(i)
                    prevWasBoundary = false
                }
            }
            idx.add(list.size)
            return idx
        }

        fun segments(list: List<Int>, idx: List<Int>): List<Pair<Int, String>> =
            (0 until idx.size - 1).map { i ->
                val part = list.subList(idx[i], idx[i + 1])
                part.size to engine.tokenizer.decode(part)
            }

        /** Last resort for a segment with no `,;:` to break on: split at spaces
         *  so no chunk exceeds the LM's per-chunk token budget. A single word
         *  longer than the cap cannot be split further and stays as-is. */
        fun splitOnSpace(text: String): List<Pair<Int, String>> {
            val words = text.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
            val out = ArrayList<Pair<Int, String>>()
            var cur = StringBuilder()
            var count = 0
            for (w in words) {
                val cand = if (cur.isEmpty()) w else "$cur $w"
                val n = engine.tokenizer.encode(cand).size
                if (count > 0 && n > PocketTts.MAX_TOKENS_PER_CHUNK) {
                    out.add(count to cur.toString())
                    cur = StringBuilder(w)
                    count = engine.tokenizer.encode(w).size
                } else {
                    cur = StringBuilder(cand)
                    count = n
                }
            }
            if (count > 0) out.add(count to cur.toString())
            return out
        }

        val sentences = segments(tokens, boundaries(tokens, engine.endTokens))
        val refined = ArrayList<Pair<Int, String>>()
        for ((n, textSeg) in sentences) {
            if (n <= PocketTts.MAX_TOKENS_PER_CHUNK) { refined.add(n to textSeg); continue }
            val sub = engine.tokenizer.encode(textSeg.trim()).toList()
            val subSegs = segments(sub, boundaries(sub, engine.fallbackTokens))
            for ((sn, st) in subSegs) {
                if (sn <= PocketTts.MAX_TOKENS_PER_CHUNK) refined.add(sn to st)
                else refined.addAll(splitOnSpace(st))
            }
        }

        val chunks = ArrayList<String>()
        var current = ""
        var count = 0
        for ((n, sentence) in refined) {
            when {
                current.isEmpty() -> { current = sentence; count = n }
                count + n > PocketTts.MAX_TOKENS_PER_CHUNK -> {
                    chunks.add(current.trim()); current = sentence; count = n
                }
                else -> { current += " $sentence"; count += n }
            }
        }
        if (current.isNotEmpty()) chunks.add(current.trim())
        return chunks
    }
}
