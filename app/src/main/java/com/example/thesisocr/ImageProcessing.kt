package com.example.thesisocr

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class ImageProcessing {
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