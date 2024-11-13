package com.example.thesisocr

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log

class ModelProcessing(private val resources: Resources) {
    private var modelVocab = loadDictionary()
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private var resizeWidth = 1280
    private var resizeHeight = 960
    private var idToProcess = 0

    data class ModelResults(
        var detectionResult: PaddleDetector.Result,
        var recognitionResult: PaddleRecognition.TextResult?,
        var recogInputBitmapList: MutableList<Bitmap>,
        var preProcessedImage: Bitmap
    )
    fun processImage(inputBitmap: Bitmap, idToProcess: Int = 0): ModelResults {
        // Set class' idToProcess to the parameter idToProcess.
        this.idToProcess = idToProcess
        // Resize dimensions for the input bitmap based on idToProcess.
        when (this.idToProcess){
            0 -> {
                resizeWidth = 1280
                resizeHeight = 960
            }
            1 -> {
                resizeWidth = 960
                resizeHeight = 1280
            }
        }
        // Resize the input bitmap.
        val resizedBitmap = ImageProcessing().rescaleBitmap(
            inputBitmap
            , resizeWidth, resizeHeight)
        ortSession = ortEnv.createSession(selectModel(1), ortSessionConfigurations())
        // Run detection on the resized bitmap.
        val detectionResult = PaddleDetector().detect(
            resizedBitmap, ortEnv, ortSession, idToProcess
        )
        // Cancel entire process if bounding box list does not meet the requirements.
        if (detectionResult.boundingBoxList.size < 2 || detectionResult.boundingBoxList.size > 25){
            return ModelResults(detectionResult, null, mutableListOf(), resizedBitmap)
        }
        ortSession.close()
        // Run recognition on the resized bitmap.
        val recogInputBitmapList = cropAndProcessBitmapList(resizedBitmap, detectionResult)
        ortSession = ortEnv.createSession(selectModel(2), ortSessionConfigurations())
        val recognitionResult =
            PaddleRecognition().recognize(recogInputBitmapList, ortEnv, ortSession, modelVocab)
        ortSession.close()
        // Return the results.
        return ModelResults(detectionResult, recognitionResult, recogInputBitmapList, resizedBitmap)
    }
    fun warmupThreads(){
        val warmupCycles = 3
        Log.d("Warm-up", "Warming up threads.")
        // Sample bitmap for warm-up.
        val warmupBitmap = ImageProcessing().rescaleBitmap(
            BitmapFactory.decodeResource(resources, R.drawable.philsys_sample),
            resizeWidth, resizeHeight
        )
        for (i in 0 until warmupCycles){
            // Run detection on warm-up bitmap.
            ortSession = ortEnv.createSession(selectModel(1), ortSessionConfigurations())
            val warmupDetection = PaddleDetector().detect(
                //ImageProcessing().processImageForDetection(warmupBitmap)
                warmupBitmap
                , ortEnv, ortSession, 0)
            ortSession.close()
            // Run recognition on warm-up bitmap.
            if (warmupDetection.boundingBoxList.isNotEmpty()){
                ortSession = ortEnv.createSession(selectModel(2), ortSessionConfigurations())
                val warmupRecognition = PaddleRecognition().recognize(
                    cropAndProcessBitmapList(warmupBitmap, warmupDetection),
                    ortEnv, ortSession, modelVocab
                )
                ortSession.close()
            }
        }
        Log.d("Warm-up", "Threads warmed up.")
    }
    private fun ortSessionConfigurations(): OrtSession.SessionOptions {
        // NOTE: NNAPI only supports models with fixed input dimensions, it will not work with models that have dynamic input dimensions.
        val sessionOptions = OrtSession.SessionOptions()
        // Set NNAPI flags.
        // val nnapiFlags = EnumSet.of(NNAPIFlags.CPU_ONLY)
        // Add NNAPI (Pass nnapiFlags as parameter if needed)
        sessionOptions.addNnapi()
        // Execution Mode and Optimization Level
        sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)
        sessionOptions.setIntraOpNumThreads(4)
        // Get settings information
        val sessionOptionInfo = sessionOptions.configEntries
        Log.d("Session Options", "Session Options: $sessionOptionInfo")
        // Turn on profiling
        // Save to profile.log at device's documents directory.
        // sessionOptions.enableProfiling(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString() + "/profile.log")
        return sessionOptions
    }
    private fun cropAndProcessBitmapList(inputBitmap: Bitmap, detectionResult: PaddleDetector.Result): MutableList<Bitmap>{
        val croppedBitmapList = PaddleDetector().cropBitmapToBoundingBoxes(inputBitmap, detectionResult.boundingBoxList)
        val preProcessedList = mutableListOf<Bitmap>()
        for (element in croppedBitmapList){
            preProcessedList.add(ImageProcessing().processImageForRecognition(element, idToProcess))
        }
        return preProcessedList
    }
    private fun selectModel(modelNum: Int): ByteArray{
        val modelPackagePath = when (modelNum) {
            // Detection
            1 -> R.raw.det_model
            // Recognition
            2 -> R.raw.en_v3_synth2_20epoch
            // Default to detection model.
            else -> R.raw.det_model
        }
        return resources.openRawResource(modelPackagePath).readBytes()
    }
    private fun loadDictionary(): List<String> {
        val inputStream = resources.openRawResource(R.raw.en_dict)
        val dictionary = mutableListOf<String>()
        inputStream.bufferedReader().useLines { lines ->
            lines.forEach {
                dictionary.add(it)
            }
        }
        return dictionary
    }
    private fun gatherModelOutputInputInfo(modelToLoad: ByteArray){
        ortSession = ortEnv.createSession(modelToLoad, OrtSession.SessionOptions())
        val inputInfo = ortSession.inputInfo
        val outputInfo = ortSession.outputInfo
        Log.d("Model Info", "Input Info: $inputInfo")
        Log.d("Model Info", "Output Info: $outputInfo")
    }
    fun debugGetModelInfo(modelSelect: Int){
        val selectedModelByteArray = selectModel(modelSelect)
        gatherModelOutputInputInfo(selectedModelByteArray)
    }
}