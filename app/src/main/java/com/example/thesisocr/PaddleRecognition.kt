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
internal class PaddleRecognition {
    data class TextResult(
        var listOfStrings: MutableList<String>,
    )
    fun recognize(listOfInputBitmaps: List<Bitmap>, ortEnvironment: OrtEnvironment, ortSession: OrtSession, modelVocab: List<String>): TextResult? {
        Log.d("PaddleRecognition", "Recognizing text.")
        Log.d("PaddleRecognition", "Batch size: ${listOfInputBitmaps.size}")
        return runModel(listOfInputBitmaps, ortSession, ortEnvironment, modelVocab)
    }
    private fun runModel(listOfInputBitmaps: List<Bitmap>, ortSession: OrtSession, ortEnvironment: OrtEnvironment, modelVocab: List<String>): TextResult? {
        val listOfStrings = mutableListOf<String>()
        val batchSize = listOfInputBitmaps.size
        // Add an additional dimension for the batch size at the beginning.
        // Convert all list Bitmaps to Float Array.
        var inputArray: Array<Array<Array<FloatArray>>> =
            Array(batchSize) { bitmapToFloatArray(listOfInputBitmaps[it]) }
        // Pad the width dimensions to the maximum width.
        inputArray = padWidthDimensions(inputArray)
        Log.d("PaddleRecognition", "Image Array Sizes: ${inputArray.size} x ${inputArray[0].size} x ${inputArray[0][0].size} x ${inputArray[0][0][0].size}")
        Log.d("PaddleRecognition", "Creating input tensor.")
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, inputArray)
        Log.d("PaddleRecognition", "Running model.")
        var recognitionInferenceTime = System.currentTimeMillis()
        val output = ortSession.run(
            Collections.singletonMap("x", inputTensor)
        )
        recognitionInferenceTime = System.currentTimeMillis() - recognitionInferenceTime
        Log.d("PaddleRecognition", "Model run successful: $recognitionInferenceTime ms.")
        Log.d("PaddleRecognition", "Processing output.")
        output.use {
            val rawOutput = output?.get(0)?.value as Array<Array<FloatArray>>
            // Array structure: rawOutput[batchSize][sequenceLength][modelVocab]
            // NOTE: batchSize is variable in this case.
            Log.d(
                "PaddleRecognition",
                "Output Array Sizes: ${rawOutput.size} x ${rawOutput[0].size} x ${rawOutput[0][0].size}"
            )
            for (i in 0 until batchSize) {
                val debugBitmap = convertArrayToBitmap(inputArray[i]) // TODO: Remove after debugging.
                val sequenceLength = rawOutput[i].size
                val sequence = mutableListOf<String>()
                for (j in 0 until sequenceLength) {
                    val maxIndex =  rawOutput[i][j].indices.maxByOrNull { rawOutput[i][j][it] } ?: -1
                    if (maxIndex in 1..94 && rawOutput[i][j][maxIndex] > 0.75f){
                        sequence.add(modelVocab[maxIndex - 1])
                    } else {
                        // TODO: Implement CTC loss handling.
                        // sequence.add(" ")
                    }
                }
                Log.d("PaddleRecognition", "Sequence $i: $sequence")
                listOfStrings.add(sequence.joinToString(""))
            }
        }
        return TextResult(listOfStrings)
    }
    private fun rescaleBitmap(bitmap: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }
    private fun bitmapToFloatArray(bitmap: Bitmap): Array<Array<FloatArray>> {
        // Rescale bitmap to height 48 while maintaining width
        val rescaledBitmap = rescaleBitmap(bitmap, bitmap.width, 48)
        // Convert bitmap to float array.
        val channels = 3
        val width = rescaledBitmap.width
        val height = rescaledBitmap.height
        val imageArray = Array(channels) { Array(width) { FloatArray(height) } }
        for (i in 0 until width) {
            for (j in 0 until height) {
                val pixel = rescaledBitmap.getPixel(i, j)
                imageArray[0][i][j] = (pixel shr 16 and 0xFF) / 255.0f
                imageArray[1][i][j] = (pixel shr 8 and 0xFF) / 255.0f
                imageArray[2][i][j] = (pixel and 0xFF) / 255.0f
            }
        }
        // Transpose where width and height are swapped.
        val transposedArray = Array(channels) { Array(height) { FloatArray(width) } }
        for (i in 0 until channels) {
            for (j in 0 until width) {
                for (k in 0 until height) {
                    transposedArray[i][k][j] = imageArray[i][j][k]
                }
            }
        }
        return transposedArray
    }
    private fun padWidthDimensions(inputArray: Array<Array<Array<FloatArray>>>): Array<Array<Array<FloatArray>>> {
        // Width is the 4th dimension while height is the 3rd dimension.
        val batchSize = inputArray.size
        var maxWidth = inputArray[0][0][0].size
        for (i in 0 until batchSize) {
            for (j in 0 until 3) {
                for (k in 0 until 48) {
                    if (inputArray[i][j][k].size > maxWidth) {
                        maxWidth = inputArray[i][j][k].size
                    }
                }
            }
        }
        val paddedArray = Array(batchSize) { Array(3) { Array(48) { FloatArray(maxWidth) } } }
        for (i in 0 until batchSize) {
            for (j in 0 until 3) {
                for (k in 0 until 48) {
                    for (l in 0 until maxWidth) {
                        if (l < inputArray[i][j][k].size) {
                            paddedArray[i][j][k][l] = inputArray[i][j][k][l]
                        } else {
                            paddedArray[i][j][k][l] = 0.0f
                        }
                    }
                }
            }
        }
        return paddedArray
    }
    // Debugging functions
    private fun convertArrayToBitmap(array: Array<Array<FloatArray>>): Bitmap {
        // Width is the 3rd dimension while height is the 2nd dimension.
        val width = array[0][0].size
        val height = array[0].size
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (i in 0 until width) {
            for (j in 0 until height) {
                val red = (array[0][j][i] * 255).toInt()
                val green = (array[1][j][i] * 255).toInt()
                val blue = (array[2][j][i] * 255).toInt()
                val pixel = 0xff shl 24 or (red shl 16) or (green shl 8) or blue
                bitmap.setPixel(i, j, pixel)
            }
        }
        return bitmap
    }
}