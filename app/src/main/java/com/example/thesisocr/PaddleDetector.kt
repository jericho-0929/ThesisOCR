package com.example.thesisocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.core.graphics.blue
import androidx.core.graphics.green
import androidx.core.graphics.red
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.Collections

internal data class Result(
    var outputBitmap: Bitmap
)

/**
 * PaddleDetector class for processing images using the PaddlePaddle model.
 */

/**
 * ONNX Model Input and Output info's shows the following:
 * Shapes are formatted as: (Batch Size, Channels, Width, Height)
 * Input consists of a Tensor of shape (Batch Size, 3, inputWidth, inputHeight).
 * Output consists of a Tensor of shape (Batch Size, 1, outputWidth, outputHeight).
 * Accordingly, this model outputs an image where any detected text boxes are highlighted.
 * Rest of the image is
 */
internal class PaddleDetector {
    fun detect(bitmap: Bitmap, ortEnvironment: OrtEnvironment, ortSession: OrtSession): Result? {
        val imageArray = convertImageToFloatArray(bitmap)
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, imageArray)
        Log.d("PaddleDetector", "Input Tensor: ${inputTensor.info}")
        return runModel(inputTensor, ortSession, bitmap)
    }
    /**  TODO: Determine cause of crash when running the model
        *   with a lower than the camera's default resolution
        *   but works with a 640 x 480 input.
     *   TODO: Implement method to merge/mask output image with input image.
     */
    private fun runModel(onnxTensor: OnnxTensor, ortSession: OrtSession, inputBitmap: Bitmap): Result? {
        val bitmapWidth = inputBitmap.width
        val bitmapHeight = inputBitmap.height
        val output = ortSession.run(
            Collections.singletonMap("x", onnxTensor)
        )
        Log.d("PaddleDetector", "Model run completed.\nHandling output.")
        output.use {
            val rawOutput = output?.get(0)?.value as Array<Array<Array<FloatArray>>>
            // Convert rawOutput to a Bitmap
            var outputImageBitmap = Bitmap.createBitmap(bitmapWidth,bitmapHeight, Bitmap.Config.ARGB_8888)
            val multiplier = -255.0f * 2
            for (i in 0 until bitmapWidth) {
                for (j in 0 until bitmapHeight) {
                    val pixelIntensity = (rawOutput[0][0][i][j] * multiplier).toInt()
                    outputImageBitmap.setPixel(i, j, Color.rgb(pixelIntensity, pixelIntensity, pixelIntensity))
                }
            }
            // Log.d("PaddleDetector", "Pixel at 0,0: ${outputImageBitmap.getPixel(0,0).red}")
            // debugSaveImage(outputImageBitmap, Environment.getExternalStorageDirectory().toString() + "/Pictures/output.jpg")
            Log.d("PaddleDetector", "Model output handled.\nPaddleDetector completed.")
            outputImageBitmap = maskInputWithOutput(inputBitmap, outputImageBitmap)
            return Result(outputImageBitmap)
        }
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
    private fun maskInputWithOutput(inputBitmap: Bitmap, outputBitmap: Bitmap): Bitmap {
        val width = inputBitmap.width
        val height = inputBitmap.height
        Log.d("PaddleDetector", "Masking input with output.")
        // Dilate the outputBitmap to increase the size of the detected text boxes
        val dilatedBitmap = imageDilation(outputBitmap)
        val maskedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        // Keep only input pixels within the yellow boxes found in the outputBitmap
        for (i in 0 until width) {
            for (j in 0 until height) {
                val outputPixel = dilatedBitmap.getPixel(i, j)
                if (outputPixel.red >= 200 && outputPixel.green >= 200 && outputPixel.blue <= 100){
                    maskedBitmap.setPixel(i, j, inputBitmap.getPixel(i, j))
                } else {
                    maskedBitmap.setPixel(i, j, Color.BLACK)
                }
            }
        }
        Log.d("PaddleDetector", "Masking completed.")
        return maskedBitmap
    }
    private fun imageDilation(inputBitmap: Bitmap): Bitmap {
        val width = inputBitmap.width
        val height = inputBitmap.height
        val inputMat = Mat()
        val dilatedMat = Mat()
        // Convert inputBitmap to grayscale
        Utils.bitmapToMat(inputBitmap, inputMat)
        Imgproc.dilate(inputMat, dilatedMat, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(10.0, 10.0)))
        // Convert dilatedMat to Bitmap
        val dilatedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dilatedMat, dilatedBitmap)
        return inputBitmap
    }
    // Debugging functions
    private fun debugSaveImage(bitmap: Bitmap, filename: String){
        val fileOutputStream = FileOutputStream(filename)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
        fileOutputStream.close()
    }
    private fun saveFloatBufferAsImage(floatBuffer: FloatBuffer, width: Int, height: Int, filename: String){
        val outputImageBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        var index = 0
        for (i in 0 until width) {
            for (j in 0 until height) {
                val red = (floatBuffer[index] * 255).toInt()
                val green = (floatBuffer[index + 1] * 255).toInt()
                val blue = (floatBuffer[index + 2] * 255).toInt()
                index += 3
                outputImageBitmap.setPixel(i, j, Color.rgb(red, blue, green))
            }
        }
        debugSaveImage(outputImageBitmap, filename)
    }
}