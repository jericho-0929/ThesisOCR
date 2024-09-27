package com.example.thesisocr

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Environment
import android.util.Log
import java.util.EnumSet

class ModelProcessing(private val resources: Resources) {
    private var modelVocab = loadDictionary()
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    // private var detSession: OrtSession = ortEnv.createSession(selectModel(1), ortSessionConfigurations())
    // private var recogSession: OrtSession = ortEnv.createSession(selectModel(2), ortSessionConfigurations())
    private val resizeWidth = 1280
    private val resizeHeight = 960

    data class ModelResults(
        var detectionResult: PaddleDetector.Result,
        var recognitionResult: PaddleRecognition.TextResult?,
        var recogInputBitmapList: MutableList<Bitmap>
    )
    fun processImage(inputBitmap: Bitmap): ModelResults {
        val resizedBitmap = ImageProcessing().rescaleBitmap(
            inputBitmap
            , resizeWidth, resizeHeight)
        // Image pre-processing.
        ortSession = ortEnv.createSession(selectModel(1), ortSessionConfigurations())
        val detectionResult = PaddleDetector().detect(resizedBitmap, ortEnv, ortSession)
        // Cancel entire process if bounding box list is less than 12 and more than 13.
        if (detectionResult.boundingBoxList.size < 6 || detectionResult.boundingBoxList.size > 25){
            return ModelResults(detectionResult, null, mutableListOf())
        }
        ortSession.close()
        val recogInputBitmapList = cropAndProcessBitmapList(resizedBitmap, detectionResult)
        ortSession = ortEnv.createSession(selectModel(2), ortSessionConfigurations())
        val recognitionResult =
            PaddleRecognition().recognize(recogInputBitmapList, ortEnv, ortSession, modelVocab)
        ortSession.close()
        return ModelResults(detectionResult, recognitionResult, recogInputBitmapList)
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
                , ortEnv, ortSession)
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
        // NOTE: NNAPI only supports models with fixed input dimensions.
        val sessionOptions = OrtSession.SessionOptions()
        // Set NNAPI flags.
        //val nnapiFlags = EnumSet.of(NNAPIFlags.CPU_ONLY)
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
        sessionOptions.enableProfiling(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString() + "/profile.log")
        return sessionOptions
    }
    private fun cropAndProcessBitmapList(inputBitmap: Bitmap, detectionResult: PaddleDetector.Result): MutableList<Bitmap>{
        val croppedBitmapList = PaddleDetector().cropBitmapToBoundingBoxes(inputBitmap, detectionResult.boundingBoxList)
        val preProcessedList = mutableListOf<Bitmap>()
        for (element in croppedBitmapList){
            preProcessedList.add(ImageProcessing().processImageForRecognition(element))
        }
        return preProcessedList
    }
    private fun selectModel(modelNum: Int): ByteArray{
        val modelPackagePath = when (modelNum) {
            // Detection
            1 -> R.raw.det_model
            // Recognition
            2 -> R.raw.en_v3_synth4_20epoch
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