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
import kotlin.time.Duration
import kotlin.time.measureTime

import com.google.gson.Gson
import java.io.File
import kotlin.math.pow
import kotlin.math.sqrt

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
// TODO: REMOVE ANY VARIABLES THAT ARE MARKED FOR DEBUGGING PURPOSES.
class PaddleDetector {
    data class Result(
        var outputMask: Bitmap,
        var outputBitmap: Bitmap,
        var boundingBoxList: List<BoundingBox>,
        var inferenceTime: Duration
    )
    data class BoundingBox(val x: Int, val y: Int, val width: Int, val height: Int)
    private var inferenceTime: Duration = Duration.ZERO

    fun detect(inputBitmap: Bitmap, ortEnvironment: OrtEnvironment, ortSession: OrtSession): Result {
        val bitmapWidth = inputBitmap.width
        val bitmapHeight = inputBitmap.height
        // Resize the inputBitmap to the model's input size.
        val resizedBitmap = ImageProcessing().rescaleBitmap(inputBitmap, bitmapWidth, bitmapHeight)
        val preprocessedBitmap = ImageProcessing().processImageForDetection(resizedBitmap)
        Log.d("PaddleDetector", "Resized Bitmap: ${resizedBitmap.width} x ${resizedBitmap.height}")
        // Split the inputArray into chunks.
        val inferenceChunks: List<Array<Array<Array<FloatArray>>>> = splitBitmapIntoChunks(resizedBitmap).map {
            convertImageToFloatArray(it)
        }
        val referenceChunks = splitBitmapIntoChunks(resizedBitmap)
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
            inferenceTime = totalInferenceTime
            Log.d("PaddleDetector", "Processing time (inc. overhead): $totalInferenceTime")
        }
        Log.d("PaddleDetector", "Inference complete.")
        val rawBitmapList = mutableListOf<Bitmap>()
        for (i in resultList.indices) {
            rawBitmapList.add(
                opening(
                    processRawOutput(resultList[i], referenceChunks[i])
                    , 5.0, 5.0
                )
            )
        }
        // Fix the output bitmaps by closing horizontal gaps.
        val fixedBitmapList = mutableListOf<Bitmap>()
        val pixelDistance = rawBitmapList[0].width / 4
        for (i in resultList.indices) {
            // First bitmap: closeHorizontalGapsRightOnly, Last bitmap: closeHorizontalGapsLeftOnly, Others: closeHorizontalGaps
            when (i) {
                0 -> fixedBitmapList.add(closeHorizontalGapsRightOnly(rawBitmapList[i], pixelDistance))
                resultList.size - 1 -> fixedBitmapList.add(closeHorizontalGapsLeftOnly(rawBitmapList[i], pixelDistance))
                else -> fixedBitmapList.add(closeHorizontalGaps(rawBitmapList[i], pixelDistance))
            }
        }
        // Stitch the output bitmaps together
        val debugOutputBitmap = ImageProcessing().convertToBitmap(
            ImageProcessing().convertBitmapToMat(
                stitchBitmapChunks(fixedBitmapList)
            )
        )
        val outputBitmap = ImageProcessing().convertToBitmap(
                ImageProcessing().convertBitmapToMat(
                    stitchBitmapChunks(fixedBitmapList)
                )
        )
        // Creation of bounding boxes from the outputBitmap.
        // Resize the outputBitmap to the original inputBitmap's size.
        val resizedOutputBitmap = ImageProcessing().processDetectionOutputMask(
            ImageProcessing().rescaleBitmap(
                outputBitmap, bitmapWidth, bitmapHeight
            )
        )
        val boundingBoxList = createBoundingBoxes(convertImageToFloatArray(convertToMonochrome(resizedOutputBitmap)), resizedOutputBitmap)
        // Render bounding boxes on the inputBitmap.
        val renderedBitmap = renderBoundingBoxes(inputBitmap, boundingBoxList)
        return Result(outputBitmap, renderedBitmap, boundingBoxList, inferenceTime)
    }
    fun detectSingle(inputBitmap: Bitmap, ortEnvironment: OrtEnvironment, ortSession: OrtSession): Result {
        // Resize bitmap
        val resizedBitmap = ImageProcessing().processImageForDetection(ImageProcessing().rescaleBitmap(inputBitmap, 1280, 960))
        // Convert to FloatArray
        val inputArray = convertImageToFloatArray(resizedBitmap)
        val normalizedArray = normalizeFloatArray(inputArray)
        val debugBitmap = convertFloatArrayToImage(normalizedArray)
        // Start inference
        val rawOutput = runModel(convertImageToFloatArray(debugBitmap), ortEnvironment, ortSession)
        // Process raw output
        val outputBitmap = processRawOutput(rawOutput, resizedBitmap)
        // Generate bounding boxes
        val boundingBoxList = createBoundingBoxes(convertImageToFloatArray(convertToMonochrome(outputBitmap)), outputBitmap)
        // Render bounding boxes on the inputBitmap.
        val renderedBitmap = renderBoundingBoxes(inputBitmap, boundingBoxList)
        // Return the result.
        return Result(outputBitmap, renderedBitmap, boundingBoxList, inferenceTime)
    }
    private fun normalizeFloatArray(inputArray: Array<Array<Array<FloatArray>>>): Array<Array<Array<FloatArray>>>{
        // Normalize the inputArray through the use of mean and standard deviation.
        // Dimensions: (Batch Size, Channels, Width, Height)
        val scale = 1/255.0f
        val width = inputArray[0][0].size
        val height = inputArray[0][0][0].size
        val normalizedArray = Array(1) { Array(3) { Array(width) { FloatArray(height) } } }
        // Mean and standard deviation values for each channel.
        val meanValues = floatArrayOf(0.615f,0.667f,0.724f)
        val stdValues = floatArrayOf(0.232f,0.240f,0.240f)
        for (k in 0 until 3) {
            for (i in 0 until width) {
                for (j in 0 until height) {
                    normalizedArray[0][k][i][j] = (inputArray[0][k][i][j] * scale - meanValues[k]) / stdValues[k]
                }
            }
        }
        return normalizedArray
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
    // Multiprocessing (coroutine) helper functions.
    // Split inputBitmap into sequential chunks.
    private fun splitBitmapIntoChunks(inputBitmap: Bitmap): List<Bitmap> {
        // Split the inputBitmap into chunks.
        val chunkList = mutableListOf<Bitmap>()
        val chunkWidth = inputBitmap.width / 4
        val chunkHeight = inputBitmap.height
        for (i in 0 until 4) {
            val chunk = Bitmap.createBitmap(inputBitmap, i * chunkWidth, 0, chunkWidth, chunkHeight)
            chunkList.add(chunk)
        }
        return chunkList
    }
    private fun processRawOutput(rawOutput: OrtSession.Result, inputBitmap: Bitmap): Bitmap {
        // Feature map from the model's output.
        val outputArray = rawOutput.get(0).value as Array<Array<Array<FloatArray>>>
        // Convert rawOutput to a Bitmap
        val outputBitmap = fourDimensionArrayToRGBBitmap(outputArray)
        // Save outputBitmap as a JSON file for debugging purposes.
        // TODO: Remove this line for final package.
        fourDimensionArrayToJSON(outputArray, "/storage/emulated/0/Download/output.json")
        return outputBitmap
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
    private fun fourDimensionArrayToJSON(array: Array<Array<Array<FloatArray>>>, filename: String) {
        val gson = Gson()
        val jsonString = gson.toJson(array)
        File(filename).writeText(jsonString)
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
    private fun convertFloatArrayToImage(floatArray: Array<Array<Array<FloatArray>>>): Bitmap {
        val width = floatArray[0][0].size
        val height = floatArray[0][0][0].size
        val channels = floatArray[0].size
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        var minVal = 0
        var maxVal = 0
        for (i in 0 until width) {
            for (j in 0 until height) {
                for (k in 0 until channels) {
                    minVal = minOf(minVal, floatArray[0][k][i][j].toInt())
                    maxVal = maxOf(maxVal, floatArray[0][k][i][j].toInt())
                }
            }
        }
        for (i in 0 until width) {
            for (j in 0 until height) {
                val red = ((floatArray[0][0][i][j] - minVal) / (maxVal - minVal) * 255).toInt()
                val green = ((floatArray[0][1][i][j] - minVal) / (maxVal - minVal) * 255).toInt()
                val blue = ((floatArray[0][2][i][j] - minVal) / (maxVal - minVal) * 255).toInt()
                val pixel = 0xff shl 24 or (red shl 16) or (green shl 8) or blue
                bitmap.setPixel(i, j, pixel)
            }
        }
        return bitmap
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
    // Debugging functions
    private fun debugSaveImage(bitmap: Bitmap, filename: String){
        val fileOutputStream = FileOutputStream(filename)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
        fileOutputStream.close()
    }
}