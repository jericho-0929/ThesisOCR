package com.example.thesisocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.util.Log
import androidx.core.graphics.blue
import androidx.core.graphics.green
import androidx.core.graphics.red
import java.io.ByteArrayOutputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections

internal data class Result(
    var outputBitmap: Bitmap
)

/**
 * PaddlePredictor class for processing images using the PaddlePaddle model.
 */

/**
 * ONNX Model Input and Output info's shows the following:
 * Shapes are formatted as: (Batch Size, Channels, Width, Height)
 * Input consists of a Tensor of shape (Batch Size, 3, inputWidth, inputHeight).
 * Output consists of a Tensor of shape (Batch Size, 1, outputWidth, outputHeight).
 */
internal class PaddlePredictor {
    fun detect(bitmap: Bitmap, ortEnvironment: OrtEnvironment, ortSession: OrtSession): Result {
        val inputTensor = convertImageToFloatTensor(bitmap, ortEnvironment)
        return runONNXModel(inputTensor, ortSession)
    }
    /**  TODO: Determine cause of crash when running the model
        *    with a lower than the camera's default resolution
        *   but works with a 640 x 480 input.
     *   TODO: Modify runONNXModel to output a list of bounding boxes.
     */
    private fun runONNXModel(onnxTensor: OnnxTensor, ortSession: OrtSession): Result {
        val output = ortSession.run(Collections.singletonMap("x", onnxTensor))
        Log.d("PaddlePredictor", "Output: ${output.size()}")
        output.use {
            val itOutput = it[0] as OnnxTensor
            val shape = itOutput.info.shape
            val width = shape[2].toInt()
            val height = shape[3].toInt()
            Log.d("PaddlePredictor", "Output width: $width, height: $height")
            val buffer = FloatArray(width * height)
            onnxTensor.floatBuffer.get(buffer)
            val outputImageBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            for (i in 0 until width) {
                for (j in 0 until height) {
                    val value = buffer[i * height + j]
                    val intVal = (value * 255).toInt()
                    outputImageBitmap.setPixel(i, j, Color.rgb(intVal, intVal, intVal))
                }
            }
            onnxTensor.close()
            return Result(outputImageBitmap)
        }
    }
    private fun convertImageToFloatTensor(bitmap: Bitmap, ortEnv: OrtEnvironment): OnnxTensor{
        Log.d("PaddlePredictor", "Converting image to float tensor")
        Log.d("PaddlePredictor", "Bitmap width: ${bitmap.width}, height: ${bitmap.height}")
        val shape = longArrayOf(1, 3, bitmap.width.toLong(), bitmap.height.toLong())
        val bufferSize = bitmap.width * bitmap.height * 3

        val buffer = ByteBuffer.allocateDirect(bufferSize * 4)
        buffer.order(ByteOrder.nativeOrder())

        val floatBuffer = buffer.asFloatBuffer()

        for (i in 0 until bitmap.width) {
            for (j in 0 until bitmap.height) {
                val color = bitmap.getPixel(i, j)
                floatBuffer.put(color.red / 255.0f)
                floatBuffer.put(color.green / 255.0f)
                floatBuffer.put(color.blue / 255.0f)
            }
        }
        floatBuffer.rewind()
        Log.d("PaddlePredictor","Converting image to float tensor completed.")
        return OnnxTensor.createTensor(ortEnv, floatBuffer, shape)
    }
    private fun debugSaveImage(bitmap: Bitmap, filename: String){
        val fileOutputStream = FileOutputStream(filename)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
        fileOutputStream.close()
    }
}