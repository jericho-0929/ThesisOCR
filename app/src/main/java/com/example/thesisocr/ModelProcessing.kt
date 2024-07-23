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
    private var ortSession: OrtSession = ortEnv.createSession(selectModel(1), ortSessionConfigurations())
    private val premadeMask = BitmapFactory.decodeResource(resources, R.drawable.philsys_mask)

    data class ModelResults (
        var detectionResult: PaddleDetector.Result,
        var recognitionResult: PaddleRecognition.TextResult
    )
    fun processImage(inputBitmap: Bitmap): ModelResults {
        val resizeWidth = 1280
        val resizeHeight = 960
        val resizedBitmap = ImageProcessing().rescaleBitmap(inputBitmap, resizeWidth, resizeHeight)
        ortSession = ortEnv.createSession(selectModel(1), ortSessionConfigurations())
        val detectionResult = PaddleDetector().detect(resizedBitmap, ortEnv, ortSession)
        ortSession.close()
        /*
        val recogInputBitmapList = PaddleDetector().createBitmapListUsingPremadeMask(
            resizedBitmap,
            ImageProcessing().rescaleBitmap(premadeMask, resizeWidth, resizeHeight)
        )
        */
        val recogInputBitmapList = cropAndProcessBitmapList(resizedBitmap, detectionResult)
        ortSession = ortEnv.createSession(selectModel(2), ortSessionConfigurations())
        val recognitionResult =
            PaddleRecognition().recognize(recogInputBitmapList, ortEnv, ortSession, modelVocab)
        ortSession.close()
        return ModelResults(detectionResult, recognitionResult)
    }
    fun warmupThreads(){
        Log.d("Warm-up", "Warming up threads.")
        // Empty bitmaps for warm-up.
        val warmupBitmap = Bitmap.createBitmap(1280, 960, Bitmap.Config.ARGB_8888)
        // Only run detection model.
        PaddleDetector().detect(warmupBitmap, ortEnv, ortSession)
        Log.d("Warm-up", "Threads warmed up.")
    }
    fun getModelInfo(modelSelect: Int){
        debugGetModelInfo(modelSelect)
    }
    private fun ortSessionConfigurations(): OrtSession.SessionOptions {
        val sessionOptions = OrtSession.SessionOptions()
        // Set NNAPI flags.
        val nnapiFlags = EnumSet.of(NNAPIFlags.CPU_DISABLED)
        // Add NNAPI
        sessionOptions.addNnapi(nnapiFlags)
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
            1 -> R.raw.det_model
            2 -> R.raw.rec_model
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
    private fun debugGetModelInfo(modelSelect: Int){
        val selectedModelByteArray = selectModel(modelSelect)
        gatherModelOutputInputInfo(selectedModelByteArray)
    }
}