package com.example.thesisocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Environment
import android.util.Log
import androidx.core.graphics.blue
import androidx.core.graphics.green
import androidx.core.graphics.red
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
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
        runModel(inputTensor, ortSession)
        return null
    }
    /**  TODO: Determine cause of crash when running the model
        *    with a lower than the camera's default resolution
        *   but works with a 640 x 480 input.
     *   TODO: Implement method to merge/mask output image with input image.
     */
    private fun runModel(onnxTensor: OnnxTensor, ortSession: OrtSession) {
        val output = ortSession.run(
            Collections.singletonMap("x", onnxTensor)
        )
        Log.d("PaddleDetector", "Model run completed.\nHandling output.")
        output.use {
            val rawOutput = output?.get(0)?.value as Array<Array<Array<FloatArray>>>
            // Convert rawOutput to a Bitmap
            val outputImageBitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
            val multiplier = -255.0f
            for (i in 0 until 640) {
                for (j in 0 until 480) {
                    val pixelIntensity = (rawOutput[0][0][i][j] * multiplier).toInt()
                    outputImageBitmap.setPixel(i, j, Color.rgb(pixelIntensity, pixelIntensity, pixelIntensity))
                }
            }
            // Log.d("PaddleDetector", "Pixel at 0,0: ${outputImageBitmap.getPixel(0,0).red}")
            // debugSaveImage(outputImageBitmap, Environment.getExternalStorageDirectory().toString() + "/Pictures/output.jpg")
            Result(outputImageBitmap)
        }
        Log.d("PaddleDetector", "Model output handled.\nPaddleDetector completed.")
    }
    private fun convertImageToFloatArray(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val floatArray = Array(1) { Array(3) { Array(640) { FloatArray(480) } } }
        for (i in 0 until 640) {
            for (j in 0 until 480) {
                val color = bitmap.getPixel(i, j)
                floatArray[0][0][i][j] = color.red / 255.0f
                floatArray[0][1][i][j] = color.green / 255.0f
                floatArray[0][2][i][j] = color.blue / 255.0f
            }
        }
        return floatArray
    }
    private fun convertImageToFloatTensor(bitmap: Bitmap, ortEnv: OrtEnvironment): OnnxTensor{
        Log.d("PaddleDetector", "Converting image to float tensor")
        Log.d("PaddleDetector", "Bitmap width: ${bitmap.width}, height: ${bitmap.height}")
        val bufferSize = bitmap.width * bitmap.height * 3

        val buffer = ByteBuffer.allocateDirect(bufferSize * 4)
        buffer.order(ByteOrder.nativeOrder())

        val floatBuffer = buffer.asFloatBuffer()
        Log.d("PaddleDetector", "Pixel at 0,0: ${bitmap.getPixel(0,0).red}")
        for (i in 0 until bitmap.width) {
            for (j in 0 until bitmap.height) {
                val color = bitmap.getPixel(i, j)
                floatBuffer.put(color.red / 255.0f)
                floatBuffer.put(color.green / 255.0f)
                floatBuffer.put(color.blue / 255.0f)
            }
        }
        floatBuffer.rewind()
        // saveFloatBufferAsImage(floatBuffer, bitmap.width, bitmap.height, Environment.getExternalStorageDirectory().toString() + "/Pictures/debugOutput.jpg")
        Log.d("PaddleDetector","Converting image to float tensor completed.")
        return OnnxTensor.createTensor(ortEnv, floatBuffer)
    }
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