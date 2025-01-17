package com.example.thesisocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import java.util.Collections
import kotlin.math.roundToInt
import kotlin.time.Duration
import kotlin.time.measureTime

/**
 * PaddleRecognition class for processing images using the PaddleOCR text recognition model.
 * Tensor input shape: (Batch Size, 3, Width, Height)
 * Tensor output shape: (Batch Indices, Sequence Length, Model's Vocabulary)
 * Model's Vocabulary is the selected _dict.txt file found in ModelProcessing.kt.
 */

class PaddleRecognition {
    data class TextResult(
        var listOfStrings: MutableList<String>,
        var inferenceTime: Duration
    )
    private val tensorInputHeight = 48
    private var inferenceTime: Duration = Duration.ZERO
    fun recognize(listOfInputBitmaps: List<Bitmap>, ortEnvironment: OrtEnvironment, ortSession: OrtSession, modelVocab: List<String>): TextResult {
        // Variables for recognition output.
        val listOfStrings = mutableListOf<String>()
        val recognitionOutput = mutableListOf<List<String>>()
        val batchSize = listOfInputBitmaps.size
        // Add an additional dimension for the batch size at the beginning.
        // Convert all list Bitmaps to Float Array.
        val inputArray: Array<Array<Array<FloatArray>>> =
            Array(batchSize) { bitmapToFloatArray(listOfInputBitmaps[it]) }
        // Pad the width dimensions to the maximum width.
        // inputArray = padWidthDimensions(inputArray)
        // Split inputArray into chunks.
        val inferenceChunks = splitIntoChunks(inputArray, 4.0)
        val toAdd: List<OrtSession.Result>
        Log.d("PaddleRecognition", "Starting recognition inference.")
        // Process each chunk in parallel using async().
        runBlocking {
            val deferredList = mutableListOf<Deferred<MutableList<OrtSession.Result>>>()
            for (chunk in inferenceChunks) {
                // Launch a coroutine for each chunk.
                val deferred = async(Dispatchers.Default) {
                    Log.d("PaddleRecognition", "Thread: ${Thread.currentThread().id}.")
                    // Chunks of one.
                    val imageChunks = chunk.chunked(1)
                    val results = mutableListOf<OrtSession.Result>()
                    for (imageToProcess in imageChunks) {
                        val result = performInference(imageToProcess, ortSession, ortEnvironment)
                        results.add(result)
                    }
                    results
                }
                // Add the deferred to the list.
                deferredList.add(deferred)
            }
            // Wait for all coroutines to finish and collect their results.
            val recognitionInferenceTime = measureTime {
                toAdd = deferredList.awaitAll().flatten()
            }
            // Add all strings to listOfStrings.
            inferenceTime = recognitionInferenceTime
            Log.d("PaddleRecognition", "Processing time (inc. overhead): $recognitionInferenceTime.")
        }
        Log.d("PaddleRecognition", "Inference completed.")
        // Process raw output to get the final list of strings.
        for (result in toAdd) {
            recognitionOutput.add(processRawOutput(result, modelVocab))
        }
        // Flatten the list of strings.
        for (element in recognitionOutput) {
            listOfStrings.addAll(element)
        }
        return TextResult(listOfStrings, inferenceTime)
    }
    // 0.615f,0.667f,0.724f
    // 0.232f,0.240f,0.240f)
    // Helper functions
    private fun rescaleBitmap(bitmap: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }
    private fun bitmapToFloatArray(bitmap: Bitmap): Array<Array<FloatArray>> {
        // Rescale bitmap to height 48 while maintaining width
        val rescaledBitmap = rescaleBitmap(bitmap, bitmap.width, tensorInputHeight)
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
    // Coroutine helper functions
    private fun performInference(chunk: List<Array<Array<FloatArray>>>, ortSession: OrtSession, ortEnvironment: OrtEnvironment): OrtSession.Result {
        // Convert chunk to Array<Array<Array<FloatArray>>>.
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, chunk.toTypedArray())
        Log.d("PaddleRecognition", "Input Tensor Info: ${inputTensor.info}")
        var output: OrtSession.Result
        val inferenceTime = measureTime {
            output = ortSession.run(Collections.singletonMap("x", inputTensor))
        }
        Log.d("PaddleRecognition", "Thread ID: ${Thread.currentThread().id}; Inference time: $inferenceTime.")
        return output
    }
    private fun processRawOutput(rawOutput: OrtSession.Result, modelVocab: List<String>): List<String> {
        val listOfStrings = mutableListOf<String>()
        // Array structure: rawOutput[batchSize][sequenceLength][modelVocab]
        val rawOutputArray = rawOutput.get(0).value as Array<Array<FloatArray>>
        val modelVocabSize = modelVocab.size
        // NOTE: batchSize is variable in this case.
        for (i in rawOutputArray.indices) {
            val sequenceLength = rawOutputArray[i].size
            val sequence = mutableListOf<String>()
            var lastChar = ""
            for (j in 0 until sequenceLength) {
                val maxIndex = rawOutputArray[i][j].indices.maxByOrNull { rawOutputArray[i][j][it] } ?: -1
                if (maxIndex in 1..modelVocabSize && rawOutputArray[i][j][maxIndex] > 0.00f) {
                    val currentChar = modelVocab[maxIndex - 1]
                    if (!(lastChar == " " && currentChar == " ")) {
                        sequence.add(currentChar)
                        lastChar = currentChar
                    }
                } else {
                    // CTC Loss Handling
                    if (maxIndex >= modelVocabSize + 1) {
                        sequence.add(" ")
                        lastChar = " "
                    }
                }
            }
            // Only add if sequence does NOT only contain whitespaces or a single character.
            if (sequence.isNotEmpty() && sequence.joinToString("").trim().length > 1) {
                listOfStrings.add(sequence.joinToString(""))
                Log.d("PaddleRecognition", "Recognized text: ${listOfStrings[i]}.")
            }
        }
        return listOfStrings
    }
    private fun splitIntoChunks(inputArray: Array<Array<Array<FloatArray>>>, numOfChunks: Double): List<List<Array<Array<FloatArray>>>> {
        // Convert inputArray to a List datatype.
        val inputList = inputArray.toList()
        // Split inputList into four parts.
        val chunkSize = (inputList.size / numOfChunks)
        Log.d("PaddleRecognition", "Chunk size: $chunkSize.")
        return inputList.chunked(chunkSize.roundToInt())
    }
}