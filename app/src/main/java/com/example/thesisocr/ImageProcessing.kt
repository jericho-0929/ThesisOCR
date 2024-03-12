package com.example.thesisocr

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class ImageProcessing {
    // Detection pre-processing functions.
    // Blacken out 25% of the image's top and 10% of the image's right sections.
    fun sectionRemoval(inputBitmap: Bitmap): Bitmap {
        // Channel count is 4.
        val inputMat = Mat()
        Utils.bitmapToMat(inputBitmap, inputMat)
        val height = inputMat.height()
        val width = inputMat.width()
        val top = (height * 0.25).toInt()
        val right = (width * 0.10).toInt()
        // Remove the specified top sections.
        for (i in 0 until top) {
            for (j in 0 until width) {
                inputMat.put(i, j, 0.0, 0.0, 0.0, 0.0)
            }
        }
        // Remove the specified right sections.
        for (i in 0 until height) {
            for (j in 0 until right) {
                inputMat.put(i, width - j - 1, 0.0, 0.0, 0.0, 0.0)
            }
        }
        return convertToBitmap(inputMat)
    }
    // Recognition pre-processing functions.
    fun processImageForRecognition(inputBitmap: Bitmap): Bitmap {
        val grayMat = convertToGrayscaleMat(inputBitmap)
        val equalizedMat = histogramEqualization(grayMat)
        val blurredMat = imageBlur(equalizedMat)
        val thresholdMat = imageThresholding(blurredMat)
        return convertToBitmap(thresholdMat)
    }
    private fun convertToGrayscaleMat(inputBitmap: Bitmap): Mat {
        val inputMat = Mat()
        Utils.bitmapToMat(inputBitmap, inputMat)
        val grayMat = Mat()
        Imgproc.cvtColor(inputMat, grayMat, Imgproc.COLOR_BGR2GRAY)
        return grayMat
    }
    private fun histogramEqualization(inputMat: Mat): Mat {
        // Use CLAHE
        val equalizedMat = Mat()
        Imgproc.createCLAHE(2.0, Size(8.0, 8.0)).apply(inputMat, equalizedMat)
        return equalizedMat
    }
    private fun imageBlur(inputMat: Mat): Mat {
        val blurredMat = Mat()
        Imgproc.GaussianBlur(inputMat, blurredMat, Size(5.0, 5.0), 0.0)
        return blurredMat
    }
    private fun imageThresholding(inputMat: Mat): Mat {
        val thresholdMat = Mat()
        Imgproc.threshold(inputMat, thresholdMat, 135.0, 255.0, Imgproc.THRESH_OTSU)
        return thresholdMat
    }
    private fun dilation(inputMat: Mat): Mat {
        val dilatedMat = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.dilate(inputMat, dilatedMat, kernel)
        return dilatedMat
    }
    private fun convertToBitmap(inputMat: Mat): Bitmap {
        val outputBitmap = Bitmap.createBitmap(inputMat.width(), inputMat.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(inputMat, outputBitmap)
        return outputBitmap
    }
}