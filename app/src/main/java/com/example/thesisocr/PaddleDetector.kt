package com.example.thesisocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import androidx.core.graphics.blue
import androidx.core.graphics.green
import androidx.core.graphics.red
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import java.io.FileOutputStream
import java.util.Collections
import kotlin.time.measureTime

/**
 * PaddleDetector class for processing images using the PaddlePaddle model.
 */

/**
 * ONNX Model Input and Output info's shows the following:
 * Shapes are formatted as: (Batch Size, Channels, Width, Height)
 * Input consists of a Tensor of shape (Batch Size, 3, inputWidth, inputHeight).
 * Output consists of a Tensor of shape (Batch Size, 1, outputWidth, outputHeight).
 *
 * Accordingly, this model outputs a feature map and an algorithm to create bounding boxes
 * based on the feature map should be created.
 */
class PaddleDetector {
    data class Result(
        var outputMask: Bitmap,
        var outputBitmap: Bitmap,
        var boundingBoxList: List<BoundingBox>
    )
    data class BoundingBox(val x: Int, val y: Int, val width: Int, val height: Int)
    fun detect(inputBitmap: Bitmap, ortEnvironment: OrtEnvironment, ortSession: OrtSession): Result {
        val bitmapWidth = inputBitmap.width
        val bitmapHeight = inputBitmap.height
        // Resize the inputBitmap to the model's input size.
        val resizedBitmap = ImageProcessing().rescaleBitmap(
            inputBitmap, bitmapWidth / 2,
            bitmapHeight / 2)
        Log.d("PaddleDetector", "Resized Bitmap: ${resizedBitmap.width} x ${resizedBitmap.height}")
        // Split the resizedBitmap into chunks.
        val inferenceChunks = splitBitmapIntoChunks(resizedBitmap, 4)
        val resultList: List<OrtSession.Result>
        Log.d("PaddleDetector", "Starting detection inference.")
        // Process each chunk in parallel using async().
        runBlocking {
            val deferredList = mutableListOf<Deferred<OrtSession.Result>>()
            for (chunk in inferenceChunks) {
                val deferred = async(Dispatchers.Default) {
                    Log.d("PaddleRecognition", "Thread: ${Thread.currentThread().id}.")
                    runModel(chunk, ortEnvironment, ortSession)
                }
                deferredList.add(deferred)
            }
            val totalInferenceTime = measureTime {
                resultList = deferredList.awaitAll()
            }
            Log.d("PaddleDetector", "Processing time (inc. overhead): $totalInferenceTime")
        }
        Log.d("PaddleDetector", "Inference complete.")
        // Fix the output bitmaps by closing horizontal gaps.
        val fixedBitmapList = mutableListOf<Bitmap>()
        val pixelDistance = 35
        for (i in resultList.indices) {
            // First bitmap: closeHorizontalGapsRightOnly, Last bitmap: closeHorizontalGapsLeftOnly, Others: closeHorizontalGaps
            when (i) {
                0 -> fixedBitmapList.add(closeHorizontalGapsRightOnly(processRawOutput(resultList[i], inferenceChunks[i]).outputBitmap, pixelDistance))
                resultList.size - 1 -> fixedBitmapList.add(closeHorizontalGapsLeftOnly(processRawOutput(resultList[i], inferenceChunks[i]).outputBitmap, pixelDistance))
                else -> fixedBitmapList.add(closeHorizontalGaps(processRawOutput(resultList[i], inferenceChunks[i]).outputBitmap, pixelDistance))
            }
        }
        // Stitch the output bitmaps together
        val outputBitmap = stitchBitmapChunks(fixedBitmapList)
        // Creation of bounding boxes from the outputBitmap.
        // Resize the outputBitmap to the original inputBitmap's size.
        val resizedOutputBitmap = ImageProcessing().rescaleBitmap(
            outputBitmap, bitmapWidth, bitmapHeight)
        val boundingBoxList = createBoundingBoxes(convertImageToFloatArray(convertToMonochrome(resizedOutputBitmap)), resizedOutputBitmap)
        // Render bounding boxes on the inputBitmap.
        val renderedBitmap = renderBoundingBoxes(inputBitmap, boundingBoxList)
        return Result(outputBitmap, renderedBitmap, boundingBoxList)
    }
    fun detectSingle(inputBitmap: Bitmap, ortEnvironment: OrtEnvironment, ortSession: OrtSession): Result {
        val resizedBitmap = ImageProcessing().rescaleBitmap(inputBitmap, 1280, 960)
        val rawOutput = runModel(resizedBitmap, ortEnvironment, ortSession)
        val outputBitmap = processRawOutput(rawOutput, resizedBitmap).outputBitmap
        val boundingBoxList = createBoundingBoxes(convertImageToFloatArray(convertToMonochrome(outputBitmap)), outputBitmap)
        val renderedBitmap = renderBoundingBoxes(inputBitmap, boundingBoxList)
        return Result(outputBitmap, renderedBitmap, boundingBoxList)
    }
    // Multiprocessing (coroutine) helper functions.
    // Split inputBitmap into sequential chunks.
    private fun splitBitmapIntoChunks(inputBitmap: Bitmap, numOfChunks: Int): List<Bitmap> {
        // Split the inputBitmap into chunks.
        val chunkList = mutableListOf<Bitmap>()
        // Ensure that the resulting bitmaps' width and height are in the 4:3 aspect ratio.
        return if (inputBitmap.width % numOfChunks != 0) {
            // Reduce numOfChunks by one and check again.
            splitBitmapIntoChunks(inputBitmap, numOfChunks - 1)
        } else {
            Log.d("PaddleDetector", "Number of chunks: $numOfChunks")
            val chunkWidth = inputBitmap.width / numOfChunks
            val chunkHeight = inputBitmap.height
            for (i in 0 until numOfChunks) {
                val chunk = Bitmap.createBitmap(inputBitmap, i * chunkWidth, 0, chunkWidth, chunkHeight)
                chunkList.add(chunk)
            }
            chunkList
        }
    }
    // Pass one chunk to the following function.
    private fun runModel(inputBitmap: Bitmap, ortEnvironment: OrtEnvironment, ortSession: OrtSession): OrtSession.Result {
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, convertImageToFloatArray(inputBitmap))
        Log.d("PaddleDetector", "Input Tensor: ${inputTensor.info}")
        var output: OrtSession.Result
        val inferenceTime = measureTime {
            output = ortSession.run(Collections.singletonMap("x", inputTensor))
        }
        Log.d("PaddleDetector", "Thread ID: ${Thread.currentThread().id}; Inference time: $inferenceTime")
        // Return the output as a Bitmap.
        return output
    }
    private fun processRawOutput(rawOutput: OrtSession.Result, inputBitmap: Bitmap): Result {
        // Feature map from the model's output.
        val outputArray = rawOutput.get(0).value as Array<Array<Array<FloatArray>>>
        // Convert rawOutput to a Bitmap
        val outputBitmap = Bitmap.createBitmap(inputBitmap.width, inputBitmap.height, Bitmap.Config.ARGB_8888)
        val multiplier = -255.0f * 2.0f
        for (i in 0 until inputBitmap.width) {
            for (j in 0 until inputBitmap.height) {
                val pixelIntensity = (outputArray[0][0][i][j] * multiplier).toInt()
                outputBitmap.setPixel(i, j, Color.rgb(pixelIntensity, pixelIntensity, pixelIntensity))
            }
        }
        return Result(outputBitmap ,outputBitmap, emptyList())
    }
    // Stitch the output bitmaps.
    private fun stitchBitmapChunks(bitmapList: List<Bitmap>): Bitmap {
        val firstBitmap = bitmapList[0]
        val outputBitmap = Bitmap.createBitmap(firstBitmap.width * bitmapList.size, firstBitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outputBitmap)
        for (i in bitmapList.indices) {
            canvas.drawBitmap(bitmapList[i], i * firstBitmap.width.toFloat(), 0f, null)
        }
        return outputBitmap
    }
    // Close horizontal gaps by extending non-black pixels closest to bitmap edges.
    private fun closeHorizontalGapsRightOnly(inputBitmap: Bitmap, pixelDistance: Int): Bitmap {
        val width = inputBitmap.width
        val height = inputBitmap.height
        // Copy inputBitmap to outputBitmap
        val outputBitmap = inputBitmap.copy(inputBitmap.config, true)
        // Extend non-black pixels if they are within specified pixels of the bitmap edges.
        for (i in 0 until width) {
            for (j in 0 until height) {
                val pixel = inputBitmap.getPixel(i, j)
                if (pixel != Color.BLACK) {
                    // Extend non-black pixels closest to the right edge.
                    if (i > width - pixelDistance) {
                        for (k in i until width) {
                            outputBitmap.setPixel(k, j, pixel)
                        }
                    }
                }
            }
        }
        return outputBitmap
    }
    private fun closeHorizontalGapsLeftOnly(inputBitmap: Bitmap, pixelDistance: Int): Bitmap {
        val width = inputBitmap.width
        val height = inputBitmap.height
        // Copy inputBitmap to outputBitmap
        val outputBitmap = inputBitmap.copy(inputBitmap.config, true)
        // Extend non-black pixels if they are within specified pixels of the bitmap edges.
        for (i in 0 until width) {
            for (j in 0 until height) {
                val pixel = inputBitmap.getPixel(i, j)
                if (pixel != Color.BLACK) {
                    // Extend non-black pixels closest to the left edge.
                    if (i < pixelDistance) {
                        for (k in 0 until i) {
                            outputBitmap.setPixel(k, j, pixel)
                        }
                    }
                }
            }
        }
        return outputBitmap
    }
    private fun closeHorizontalGaps(inputBitmap: Bitmap, pixelDistance: Int): Bitmap {
        val rightBitmap = closeHorizontalGapsRightOnly(inputBitmap, pixelDistance)
        return closeHorizontalGapsLeftOnly(rightBitmap, pixelDistance)
    }
    // Post-processing helper functions.
    private fun createBoundingBoxes(rawOutput: Array<Array<Array<FloatArray>>>, inputBitmap: Bitmap): List<BoundingBox> {
        // Create bounding boxes from the raw output of the model.
        val boundingBoxes = mutableListOf<BoundingBox>()
        // Get the dimensions of the input bitmap.
        val width = inputBitmap.width
        val height = inputBitmap.height
        // Create a 2D array to keep track of visited pixels.
        val visitedPixels = Array(width) { BooleanArray(height) }
        // Iterate through the raw output and create bounding boxes for non-black pixels.
        for (i in 0 until width) {
            for (j in 0 until height) {
                if (!visitedPixels[i][j] && rawOutput[0][0][i][j] > 0.0f) {
                    // Initialize bounding box coordinates
                    var minX = i
                    var minY = j
                    var maxX = i
                    var maxY = j
                    // Depth-first search to find contiguous white pixels
                    val stack = mutableListOf<Pair<Int, Int>>()
                    stack.add(Pair(i, j))
                    visitedPixels[i][j] = true
                    while (stack.isNotEmpty()) {
                        val (x, y) = stack.removeAt(stack.size - 1)
                        // Update bounding box coordinates
                        minX = minOf(minX, x)
                        minY = minOf(minY, y)
                        maxX = maxOf(maxX, x)
                        maxY = maxOf(maxY, y)
                        // Check neighboring pixels
                        for ((dx, dy) in listOf(-1 to 0, 1 to 0, 0 to -1, 0 to 1)) {
                            val newX = x + dx
                            val newY = y + dy
                            if (newX in 0 until width && newY in 0 until height &&
                                !visitedPixels[newX][newY] && rawOutput[0][0][newX][newY] > 0.0f
                            ) {
                                stack.add(Pair(newX, newY))
                                visitedPixels[newX][newY] = true
                            }
                        }
                    }
                    // Create bounding box for the contiguous white region
                    boundingBoxes.add(BoundingBox(minX - 10, minY - 10, maxX - minX + 35, maxY - minY + 25))
                }
            }
        }
        // Remove small bounding boxes
        val minBoxWidth = 50
        boundingBoxes.removeIf { it.width < minBoxWidth }
        // Add results to the boundingBoxes list
        return boundingBoxes.distinct()
    }
    private fun renderBoundingBoxes(inputBitmap: Bitmap, boundingBoxes: List<BoundingBox>): Bitmap {
        val outputBitmap = inputBitmap
        val canvas = Canvas(outputBitmap)
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2.0f
        for (box in boundingBoxes) {
            val rect = Rect(box.x, box.y, box.x + box.width, box.y + box.height)
            canvas.drawRect(rect, paint)
        }
        return outputBitmap
    }
    private fun convertImageToFloatArray(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val width = bitmap.width
        val height = bitmap.height
        val floatArray = Array(1) { Array(3) { Array(width) { FloatArray(height) } } }
        for (i in 0 until width) {
            for (j in 0 until height) {
                val color = bitmap.getPixel(i, j)
                floatArray[0][0][i][j] = color.red / 255.0f
                floatArray[0][1][i][j] = color.green / 255.0f
                floatArray[0][2][i][j] = color.blue / 255.0f
            }
        }
        return floatArray
    }
    private fun convertToMonochrome(bitmap: Bitmap): Bitmap {
        val result = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val saturationMatrix = ColorMatrix()
        saturationMatrix.setSaturation(0f)
        val paint = Paint().apply {
            colorFilter = ColorMatrixColorFilter(saturationMatrix)
        }
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return result
    }
    fun cropBitmapToBoundingBoxes(inputBitmap: Bitmap, boundingBoxList: List<BoundingBox>): List<Bitmap> {
        // BoundingBox variables: x, y, width, height
        val croppedBitmapList = mutableListOf<Bitmap>()
        for (boundingBox in boundingBoxList){
            val croppedBitmap = Bitmap.createBitmap(inputBitmap, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height)
            croppedBitmapList.add(croppedBitmap)
        }
        return croppedBitmapList
    }
    // Debugging functions
    private fun debugSaveImage(bitmap: Bitmap, filename: String){
        val fileOutputStream = FileOutputStream(filename)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
        fileOutputStream.close()
    }
}