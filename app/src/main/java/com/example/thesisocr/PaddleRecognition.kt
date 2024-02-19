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

/**
 * NOTE: PaddlePaddleOCR GitHub discussions
 */
// TODO: Determine image resolution that works.
internal data class textResults(
    var listOfStringConfidence: MutableList<Pair<String,Float>>,
)
internal class PaddleRecognition {
    val modelVocab = getModelVocabFromResources()
    fun recognize(listOfInputBitmaps: List<Bitmap>, ortEnvironment: OrtEnvironment, ortSession: OrtSession): Result? {
        Log.d("PaddleRecognition", "Recognizing text.")
        Log.d("PaddleRecognition", "Batch size: ${listOfInputBitmaps.size}")
        return runModel(listOfInputBitmaps, ortSession, ortEnvironment)
    }
    private fun runModel(listOfInputBitmaps: List<Bitmap>, ortSession: OrtSession, ortEnvironment: OrtEnvironment): Result? {
        val batchSize = listOfInputBitmaps.size
        // Add an additional dimension for the batch size at the beginning.
        // Convert all list Bitmaps to Float Array
        val inputArray: Array<Array<Array<FloatArray>>> = Array(batchSize) { bitmapToFloatArray(listOfInputBitmaps[it]) }
        // Transpose Width and Height to Height and Width

        Log.d("PaddleRecognition", "Image Array Sizes: ${imageArray.size} x ${imageArray[0].size} x ${imageArray[0][0].size} x ${imageArray[0][0][0].size}")
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, imageArray)
        val output = ortSession.run(
            Collections.singletonMap("x", inputTensor)
        )
        output.use {
            val rawOutput = output?.get(0)?.value as Array<Array<FloatArray>>
            // Array structure: rawOutput[batchSize][sequenceLength][modelVocab]
            // NOTE: batchSize is variable in this case.
            Log.d("PaddleRecognition", "Output Array Sizes: ${rawOutput.size} x ${rawOutput[0].size} x ${rawOutput[0][0].size}")
        }
        return null
    }
    private fun bitmapToFloatArray(bitmap: Bitmap): Array<Array<FloatArray>> {
        val channels = 3
        val width = bitmap.width
        val height = bitmap.height
        val imageArray = Array(channels) { Array(width) { FloatArray(height) } }
        for (i in 0 until width) {
            for (j in 0 until height) {
                val pixel = bitmap.getPixel(i, j)
                imageArray[0][i][j] = (pixel shr 16 and 0xFF) / 255.0f
                imageArray[1][i][j] = (pixel shr 8 and 0xFF) / 255.0f
                imageArray[2][i][j] = (pixel and 0xFF) / 255.0f
            }
        }
        return imageArray
    }
    private fun transposeWidthHeightToHeightWidth(inputArray: Array<Array<FloatArray>>): Array<Array<FloatArray>> {
        // Transpose only the last two dimensions with each other.
        // Dimensions: Channels, Width, Height
        val transposedArray = Array(inputArray.size) { Array(inputArray[0][0].size) { FloatArray(inputArray[0].size) } }
        for (i in inputArray.indices) {
            for (j in inputArray[0].indices) {
                for (k in inputArray[0][0].indices) {
                    transposedArray[i][k][j] = inputArray[i][j][k]
                }
            }
        }
        return transposedArray
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