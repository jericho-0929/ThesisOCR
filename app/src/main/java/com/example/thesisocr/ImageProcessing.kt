package com.example.thesisocr

import android.graphics.Bitmap
import androidx.core.graphics.blue
import androidx.core.graphics.green
import androidx.core.graphics.red
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class ImageProcessing {
    // General image processing functions.
    fun rescaleBitmap(inputBitmap: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(inputBitmap, newWidth, newHeight, true)
    }
    fun convertBitmapToMat(inputBitmap: Bitmap): Mat {
        val inputMat = Mat()
        Utils.bitmapToMat(inputBitmap, inputMat)
        Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_RGBA2RGB)
        return inputMat
    }
    // Detection pre-processing & post-processing functions.
    fun processDetectionOutputMask(inputBitmap: Bitmap): Bitmap {
        // Convert the bitmap to a Mat.
        val inputMat = convertBitmapToMat(inputBitmap)
        return convertToBitmap(dilation(inputMat, 3.0, 3.0))
    }
    fun processImageForDetection(inputBitmap: Bitmap): Bitmap {
        val blurredMat = imageBlur(convertBitmapToMat(inputBitmap))
        return convertToBitmap(
                blurredMat
        )
    }
    // Recognition pre-processing functions.
    fun processImageForRecognition(inputBitmap: Bitmap): Bitmap {
        val thresholdMat = imageThresholding(convertToGrayscaleMat(inputBitmap))
        // Blur twice.
        val blurredMat = imageBlur(thresholdMat)
        val openedMat = opening(blurredMat, 3.0, 3.0)
        return convertToBitmap(imageBlur(openedMat))
    }
    private fun convertToGrayscaleMat(inputBitmap: Bitmap): Mat {
        val inputMat = Mat()
        Utils.bitmapToMat(inputBitmap, inputMat)
        val grayMat = Mat()
        Imgproc.cvtColor(inputMat, grayMat, Imgproc.COLOR_BGR2GRAY)
        return grayMat
    }
    private fun imageBlur(inputMat: Mat): Mat {
        val blurredMat = Mat()
        Imgproc.bilateralFilter(inputMat, blurredMat, 5, 75.0, 75.0)
        return blurredMat
    }
    private fun imageThresholding(inputMat: Mat): Mat {
        val thresholdMat = Mat()
        Imgproc.adaptiveThreshold(inputMat, thresholdMat, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 95, 47.0)
        return thresholdMat
    }
    private fun dilation(inputMat: Mat, x: Double, y: Double): Mat {
        val dilatedMat = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(x, y))
        Imgproc.dilate(inputMat, dilatedMat, kernel)
        return dilatedMat
    }
    fun opening(inputMat: Mat, x: Double, y: Double): Mat {
        val openedMat = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(x,y))
        Imgproc.morphologyEx(inputMat, openedMat, Imgproc.MORPH_OPEN, kernel)
        return openedMat
    }
    fun convertToBitmap(inputMat: Mat): Bitmap {
        val outputBitmap = Bitmap.createBitmap(inputMat.width(), inputMat.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(inputMat, outputBitmap)
        return outputBitmap
    }
    fun convertImageToFloatArray(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
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
}