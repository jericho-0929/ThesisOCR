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
import android.os.Environment
import android.util.Log
import androidx.core.graphics.blue
import androidx.core.graphics.green
import androidx.core.graphics.red
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
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
        var outputBitmap: Bitmap,
        var boundingBoxList: List<BoundingBox>
    )
    data class BoundingBox(val x: Int, val y: Int, val width: Int, val height: Int)
    fun detect(bitmap: Bitmap, ortEnvironment: OrtEnvironment, ortSession: OrtSession): Result? {
        val imageArray = convertImageToFloatArray(bitmap)
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, imageArray)
        Log.d("PaddleDetector", "Input Tensor: ${inputTensor.info}")
        return runModel(ortEnvironment, ortSession, bitmap)
    }
    /**  TODO: Determine cause of crash when running the model
        *   with a lower than the camera's default resolution
        *   but works with a 640 x 480 input.
     *   TODO: Implement method to merge/mask output image with input image.
     */
    private fun runModel(ortEnvironment: OrtEnvironment, ortSession: OrtSession, inputBitmap: Bitmap): Result? {
        val bitmapWidth = inputBitmap.width
        val bitmapHeight = inputBitmap.height
        var output: OrtSession.Result
        val resizedBitmap = Bitmap.createScaledBitmap(
            inputBitmap, bitmapWidth / 2,
            bitmapHeight / 2, true)
        // Convert bitmap to array.
        val imageArray = convertImageToFloatArray(resizedBitmap)
        // Create input tensor.
        val onnxTensor = OnnxTensor.createTensor(ortEnvironment, imageArray)
        // Run the model.
        val detectionInferenceTime = measureTime {
            output = ortSession.run(
                Collections.singletonMap("x", onnxTensor)
            )
        }
        Log.d("PaddleDetector", "Detection Model Runtime: $detectionInferenceTime")
        Log.d("PaddleDetector", "Model run completed.\nHandling output.")
        output.use {
            // Feature map from the model's output.
            val rawOutput = output?.get(0)?.value as Array<Array<Array<FloatArray>>>
            // Convert rawOutput to a Bitmap
            var outputImageBitmap = Bitmap.createBitmap(
                resizedBitmap.width, resizedBitmap.height, Bitmap.Config.ARGB_8888
            )
            val multiplier = -255.0f * 2.0f
            for (i in 0 until resizedBitmap.width) {
                for (j in 0 until resizedBitmap.height) {
                    val pixelIntensity = (rawOutput[0][0][i][j] * multiplier).toInt()
                    outputImageBitmap.setPixel(
                        i, j, Color.rgb(pixelIntensity, pixelIntensity, pixelIntensity)
                    )
                }
            }
            // Upsize the outputImageBitmap to the original size.
            outputImageBitmap = Bitmap.createScaledBitmap(outputImageBitmap, bitmapWidth, bitmapHeight, true)
            // Blacken out specified sections of the outputImageBitmap.
            outputImageBitmap = ImageProcessing().sectionRemoval(outputImageBitmap)
            // Convert outputImageBitmap back to array.
            val outputArray = convertImageToFloatArray(outputImageBitmap)
            // Create bounding boxes from outputArray.
            val boundingBoxes = createBoundingBoxes(outputArray, inputBitmap)
            Log.d("PaddleDetector", "Bounding boxes created.")
            Log.d("PaddleDetector", "Model output handled.\nPaddleDetector completed.")
            val boundingBoxesImage = renderBoundingBoxes(inputBitmap, boundingBoxes)
            return Result(boundingBoxesImage, boundingBoxes)
        }
    }
    private fun createBoundingBoxes(rawOutput: Array<Array<Array<FloatArray>>>, inputBitmap: Bitmap): List<BoundingBox> {
        // Create bounding boxes from the raw output of the model.
        val boundingBoxes = mutableListOf<BoundingBox>()
        // Get the dimensions of the input bitmap.
        val width = inputBitmap.width
        val height = inputBitmap.height
        // Create a 2D array to keep track of visited pixels.
        val visitedPixels = Array(width) { BooleanArray(height) }
        val threshold = 1E-6
        // Iterate through the raw output and create bounding boxes for pixels with intensity above a threshold.
        for (i in 0 until width) {
            for (j in 0 until height) {
                if (!visitedPixels[i][j] && rawOutput[0][0][i][j] > threshold) {
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
                                !visitedPixels[newX][newY] && rawOutput[0][0][newX][newY] > threshold
                            ) {
                                stack.add(Pair(newX, newY))
                                visitedPixels[newX][newY] = true
                            }
                        }
                    }

                    // Create bounding box for the contiguous white region
                    boundingBoxes.add(BoundingBox(minX - 10, minY - 10, maxX - minX + 20, maxY - minY + 20))
                }
            }
        }
        // Remove small bounding boxes
        val minBoxWidth = 80
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
    private fun maskInputWithOutput(inputBitmap: Bitmap, outputBitmap: Bitmap): Bitmap {
        val width = inputBitmap.width
        val height = inputBitmap.height
        Log.d("PaddleDetector", "Masking input with output.")
        // Dilate the outputBitmap to increase the size of the detected text boxes
        val dilatedBitmap = outputBitmap
        val maskedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        // Keep only input pixels that overlap the non-black pixels of dilatedBitmap
        for (i in 0 until width) {
            for (j in 0 until height) {
                val inputPixel = inputBitmap.getPixel(i, j)
                val outputPixel = dilatedBitmap.getPixel(i, j)
                if (outputPixel != Color.BLACK) {
                    maskedBitmap.setPixel(i, j, inputPixel)
                } else {
                    maskedBitmap.setPixel(i, j, Color.BLACK)
                }
            }
        }
        Log.d("PaddleDetector", "Masking completed.")
        return maskedBitmap
    }
    private fun enlargeNonBlackBoxes(bitmap: Bitmap, scaleFactor: Float): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        // Create a new bitmap with the same dimensions as the original
        val enlargedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(enlargedBitmap)
        val paint = Paint()

        for (x in 0 until width) {
            for (y in 0 until height) {
                val pixel = bitmap.getPixel(x, y)
                if (pixel != Color.BLACK) {
                    // Enlarge non-black pixels
                    val newX = (x * scaleFactor).toInt()
                    val newY = (y * scaleFactor).toInt()
                    canvas.drawPoint(newX.toFloat(), newY.toFloat(), paint)
                }
            }
        }

        return enlargedBitmap
    }
    private fun imageDilation(inputBitmap: Bitmap): Bitmap {
        val width = inputBitmap.width
        val height = inputBitmap.height
        val inputMat = Mat()
        val dilatedMat = Mat()
        // Convert inputBitmap to grayscale
        Utils.bitmapToMat(inputBitmap, inputMat)
        Imgproc.dilate(inputMat, dilatedMat, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(20.0, 20.0)))
        // Convert dilatedMat to Bitmap
        val dilatedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dilatedMat, dilatedBitmap)
        return inputBitmap
    }
    fun cropBitmapToBoundingBoxes(inputBitmap: Bitmap, boundingBoxList: List<PaddleDetector.BoundingBox>): List<Bitmap> {
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