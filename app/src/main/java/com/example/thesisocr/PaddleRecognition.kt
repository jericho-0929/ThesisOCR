package com.example.thesisocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.util.Log
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
internal data class textResults(
    var listOfStrings: Array<String>
)
internal class PaddleRecognition {
    fun recognize(bitmap: Bitmap, ortEnvironment: OrtEnvironment, ortSession: OrtSession): Array<String>? {
        return runModel(bitmap, ortSession, ortEnvironment)
    }
    private fun runModel(bitmap: Bitmap, ortSession: OrtSession, ortEnvironment: OrtEnvironment): Array<String>? {
        val imageArray = bitmapToFloatArray(bitmap)
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, imageArray)
        Log.d("PaddleRecognition", "Input Tensor: ${inputTensor.info}")
        val output = ortSession.run(
            mapOf("input_1" to inputTensor)
        )
        output.use {
            val rawOutput = output?.get(0)?.value as Array<Array<FloatArray>>
            val results = mutableListOf<String>()
            for (i in 0 until rawOutput.size) {
                val result = rawOutput[i]
                val resultString = result.joinToString(separator = "") { it.toString() }
                results.add(resultString)
            }
            return null
        }
    }
    private fun bitmapToFloatArray(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val channels = 3
        val batchSize = 1
        val width = bitmap.width
        val height = bitmap.height
        val floatArray = Array(batchSize) { Array(channels) { Array(width) { FloatArray(height) } } }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = bitmap.getPixel(x, y)
                floatArray[0][0][x][y] = (pixel shr 16 and 0xFF) / 255.0f // red
                floatArray[0][1][x][y] = (pixel shr 8 and 0xFF) / 255.0f // green
                floatArray[0][2][x][y] = (pixel and 0xFF) / 255.0f // blue
            }
        }
        return floatArray
    }
}