package com.example.thesisocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.util.Log
import java.util.Collections

/**
 * PaddleRecognition class for processing images using the PaddleOCR's text recognition model.
 * Tensor input shape: (Batch Size, 3, Width, Height)
 * Tensor output shape: (Batch Indices, Sequence Length, Model's Vocabulary)
 * Model's Vocabulary: 97 classes
 * 26 uppercase + 26 lowercase + 10 digits + 33 special + 1 space + 1 CTC loss = 97 classes
 * Refer to the en_dict.txt file found in this project's raw folder.
 */

// TODO: Replace rec_model.onnx with version wth fixed input shape.
internal data class textResults(
    var listOfStrings: Array<String>
)
internal class PaddleRecognition {
    val modelVocab = getModelVocabFromResources()
    fun recognize(bitmap: Bitmap, ortEnvironment: OrtEnvironment, ortSession: OrtSession): Result? {
        Log.d("PaddleRecognition", "Recognizing text.")
        Log.d("PaddleRecognition", "Bitmap size: ${bitmap.width} x ${bitmap.height}")
        return runModel(bitmap, ortSession, ortEnvironment)
    }
    private fun runModel(bitmap: Bitmap, ortSession: OrtSession, ortEnvironment: OrtEnvironment): Result? {
        val imageArray = bitmapToFloatArray(bitmap)
        Log.d("PaddleRecognition", "Image Array Sizes: ${imageArray.size} x ${imageArray[0].size} x ${imageArray[0][0].size} x ${imageArray[0][0][0].size}")
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, imageArray)
        val output = ortSession.run(
            Collections.singletonMap("x", inputTensor)
        )
        output.use {
            val rawOutput = output?.get(0)?.value as Array<Array<FloatArray>>
            // Array structure: rawOutput[batchSize][sequenceLength][modelVocab]
            // batchSize expected to be 1
            Log.d("PaddleRecognition", "rawOutput: ${rawOutput[0][0][0]}")
            Log.d("PaddleRecognition", "rawOutput: ${rawOutput[0][0][1]}")
            Log.d("PaddleRecognition", "rawOutput: ${rawOutput[0][0][2]}")
            val listOfStrings = mutableListOf<String>()

        }
        return null
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
    private fun getModelVocabFromResources(): List<String> {
        val vocab = mutableListOf<String>()
        val inputStream = this.javaClass.classLoader?.getResourceAsStream("raw/en_dict.txt")
        inputStream?.bufferedReader()?.useLines { lines ->
            lines.forEach {
                vocab.add(it)
            }
        }
        return vocab
    }
    // Debugging functions
}