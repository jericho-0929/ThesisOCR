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
import java.util.Collections
import kotlin.time.Duration
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
 *
 * NOTE: Input Width and Height should be a multiple of 32.
 */

class PaddleDetector {
    data class Result(
        var outputMask: Bitmap,
        var outputBitmap: Bitmap,
        var boundingBoxList: List<BoundingBox>,
        var inferenceTime: Duration
    )
    data class BoundingBox(val x: Int, val y: Int, val width: Int, val height: Int)
    private var inferenceTime: Duration = Duration.ZERO

    fun detect(inputBitmap: Bitmap, ortEnvironment: OrtEnvironment, ortSession: OrtSession, idToProcess: Int = 0): Result {
        // Resize the inputBitmap to the model's input size.
        val bitmapWidth = inputBitmap.width
        val bitmapHeight = inputBitmap.height
        // Resize the inputBitmap to the model's input size.
        val resizedBitmap = ImageProcessing().rescaleBitmap(
            ImageProcessing().processImageForDetection(inputBitmap),
            //inputBitmap,
            bitmapWidth, bitmapHeight
        )
        Log.d("PaddleDetector", "Resized Bitmap: ${resizedBitmap.width} x ${resizedBitmap.height}")
        val outputBitmap: Bitmap
        // Check if runParallel is true.
        Log.d("PaddleDetector", "Processing ID type: $idToProcess")
        val inputArray = convertImageToFloatArray(resizedBitmap)

        // Run the model on the inputArray.
        val rawOutput: OrtSession.Result
        val totalInferenceTime = measureTime {
            rawOutput = runModel(inputArray, ortEnvironment, ortSession)
        }

        // Process the raw output to create the outputBitmap.
        outputBitmap = opening(processRawOutput(rawOutput), 5.0, 5.0)
        inferenceTime = totalInferenceTime

        // Creation of bounding boxes from the outputBitmap.
        // Resize the outputBitmap to the original inputBitmap size.
        val resizedOutputBitmap = ImageProcessing().processDetectionOutputMask(
            ImageProcessing().rescaleBitmap(
                outputBitmap, bitmapWidth, bitmapHeight
            )
        )
        // Generate bounding boxes from the outputBitmap.
        val boundingBoxList = filterBoundingBoxList(
                createBoundingBoxes(
                    convertImageToFloatArray(convertToMonochrome(resizedOutputBitmap)),
                    resizedOutputBitmap
                ),
                idToProcess
            )
        // Render bounding boxes on the inputBitmap.
        val renderedBitmap = renderBoundingBoxes(resizedBitmap, boundingBoxList)
        return Result(outputBitmap, renderedBitmap, boundingBoxList, inferenceTime)
    }
    // Pass one chunk to the following function.
    private fun runModel(inputArray: Array<Array<Array<FloatArray>>>, ortEnvironment: OrtEnvironment, ortSession: OrtSession): OrtSession.Result {
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, inputArray)
        Log.d("PaddleDetector", "Input Tensor: ${inputTensor.info}")
        var output: OrtSession.Result
        val inferenceTime = measureTime {
            output = ortSession.run(Collections.singletonMap("x", inputTensor))
        }
        Log.d("PaddleDetector", "Thread ID: ${Thread.currentThread().id}; Inference time: $inferenceTime")
        // Return the output as a Bitmap.
        return output
    }

    private fun processRawOutput(rawOutput: OrtSession.Result): Bitmap {
        // Feature map from the model's output.
        val outputArray = rawOutput.get(0).value as Array<Array<Array<FloatArray>>>
        // Convert rawOutput to a Bitmap
        return fourDimensionArrayToRGBBitmap(outputArray)
    }
    private fun fourDimensionArrayToRGBBitmap(array: Array<Array<Array<FloatArray>>>): Bitmap {
        val width = array[0][0].size
        val height = array[0][0][0].size
        // Base conversion on the minimum and maximum values of the array.
        val outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        var minVal = Float.MAX_VALUE
        var maxVal = Float.MIN_VALUE
        // Find the minimum and maximum values in the array.
        for (i in 0 until width) {
            for (j in 0 until height) {
                minVal = minOf(minVal, array[0][0][i][j])
                maxVal = maxOf(maxVal, array[0][0][i][j])
            }
        }
        // Convert the array to a Bitmap.
        for (i in 0 until width) {
            for (j in 0 until height) {
                val pixelIntensity = ((array[0][0][i][j] - minVal) / (maxVal - minVal) * 255).toInt()
                outputBitmap.setPixel(i, j, Color.rgb(pixelIntensity, pixelIntensity, pixelIntensity))
            }
        }
        return outputBitmap
    }
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
                            outputBitmap.setPixel(k, j, Color.WHITE)
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
                            outputBitmap.setPixel(k, j, Color.WHITE)
                        }
                    }
                }
            }
        }
        return outputBitmap
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
                    boundingBoxes.add(BoundingBox(minX - 15, minY - 10, maxX - minX + 35, maxY - minY + 25))
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
        val canvas = Canvas(inputBitmap)
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2.0f
        for (box in boundingBoxes) {
            val rect = Rect(box.x, box.y, box.x + box.width, box.y + box.height)
            canvas.drawRect(rect, paint)
        }
        return inputBitmap
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
    // Detection-exclusive image processing functions
    private fun opening(inputBitmap: Bitmap, x: Double, y: Double): Bitmap {
        val inputMat = ImageProcessing().convertBitmapToMat(inputBitmap)
        val outputMat = ImageProcessing().opening(inputMat, x, y)
        return ImageProcessing().convertToBitmap(outputMat)
    }
    // Filter bounding box list.
    private fun filterBoundingBoxList(inputBoundingBoxList: List<BoundingBox>, idToProcess: Int): List<BoundingBox>{
        return when (idToProcess){
            0 -> filterForPhilSysID(inputBoundingBoxList)
            1 -> filterForDLSU(inputBoundingBoxList)
            else -> inputBoundingBoxList
        }
    }
    private fun filterForDLSU(inputBoundingBoxList: List<BoundingBox>): List<BoundingBox> {
        // Filter bounding box lost for De La Salle University ID.
        val trimmedBoundingBoxList = mutableListOf<BoundingBox>()
        val inputMutableList = mutableListOf<BoundingBox>()
        // Converting input list into mutable list
        for (boundingBox in inputBoundingBoxList) {
            inputMutableList.add(boundingBox)
        }
        // Only get bounding boxes past the 1/4th of the image.
        inputMutableList.sortBy { it.y }
        val midBoundingX = inputMutableList[inputMutableList.size / 4].x
        val midBoundingY = inputMutableList[inputMutableList.size / 2].y
        for (boundingBox in inputMutableList) {
            if (boundingBox.x > midBoundingX && boundingBox.y > midBoundingY) {
                trimmedBoundingBoxList.add(boundingBox)
            }
        }
        return trimmedBoundingBoxList
    }
    private fun filterForPhilSysID(inputBoundingBoxList: List<BoundingBox>): List<BoundingBox> {
        // Filter bounding box list for Philippine Identification System ID.
        val trimmedBoundingBoxList = mutableListOf<BoundingBox>()
        val inputMutableList = mutableListOf<BoundingBox>()

        // Converting input list into mutable list
        for (boundingBox in inputBoundingBoxList) {
            inputMutableList.add(boundingBox)
        }
        // Sort by horizontal coordinates
        inputMutableList.sortBy { it.x }
        val minimumBoundingX = inputMutableList[0].x
        var midBoundingX = inputMutableList[0].y
        if (inputMutableList.size > 10) {
            midBoundingX = inputMutableList[10].x
        } else {
            return trimmedBoundingBoxList
        }
        // Iterate through list.
        for (boundingBox in inputMutableList) {
            // Only add if bounding box's horizontal coordinate is:
            // Found within +20 pixels of the 3rd horizontal bounding box
            if (boundingBox.x < minimumBoundingX + 75) {
                trimmedBoundingBoxList.add(boundingBox)
            } // Found within +- 20 pixels of central bounding box group
            else if (midBoundingX - 10 < boundingBox.x && boundingBox.x < midBoundingX + 10) {
                trimmedBoundingBoxList.add(boundingBox)
            }
        }
        // Sort by y-coordinates in ascending order.
        trimmedBoundingBoxList.sortBy { it.y }
        // Remove labels by removing bounding boxes after it if the y-coordinate difference is within 30 to 55 pixels.
        // Only iterate bounding boxes whose x-coordinate is within 128 pixels from the midpoint.
        var i = 1
        while (i < trimmedBoundingBoxList.size - 2) {
            val previousBox = trimmedBoundingBoxList[i - 1]
            val currentBox = trimmedBoundingBoxList[i]
            val nextBox = trimmedBoundingBoxList[i + 1]
            if (midBoundingX - 10 < currentBox.x && currentBox.x < midBoundingX + 10) {
                // Remove PII Labels
                if (currentBox.y - previousBox.y > nextBox.y - currentBox.y) {
                    trimmedBoundingBoxList.remove(currentBox)
                } else if (currentBox.width > previousBox.width && currentBox.width > nextBox.width) {
                    trimmedBoundingBoxList.remove(currentBox)
                }
            }
            i += 1
        }
        i = 4
        while (i < trimmedBoundingBoxList.size - 1) {
            val currentBox = trimmedBoundingBoxList[i]
            val nextBox = trimmedBoundingBoxList[i + 1]
            if (currentBox.x < minimumBoundingX + 75) {
                if (currentBox.height * currentBox.width < nextBox.height * nextBox.width) {
                    trimmedBoundingBoxList.remove(currentBox)
                    break
                }
            }
            i += 1
        }
        return trimmedBoundingBoxList
    }
}