package com.example.thesisocr

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class ImageProcessing {
    fun processImageForRecognition(inputBitmap: Bitmap): Bitmap {
        val grayMat = convertToGrayscaleMat(inputBitmap)
        val equalizedMat = histogramEqualization(grayMat)
        val blurredMat = gaussianBlur(equalizedMat)
        val thresholdMat = blurredMat
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
        val equalizedMat = Mat()
        Imgproc.equalizeHist(inputMat, equalizedMat)
        return equalizedMat
    }
    private fun gaussianBlur(inputMat: Mat): Mat {
        val blurredMat = Mat()
        Imgproc.medianBlur(inputMat, blurredMat, 3)
        return blurredMat
    }
    private fun imageThresholding(inputMat: Mat): Mat {
        val thresholdMat = Mat()
        Imgproc.threshold(inputMat, thresholdMat, 0.0, 255.0, Imgproc.THRESH_OTSU)
        return thresholdMat
    }
    private fun openingOperation(inputMat: Mat): Mat {
        val morphMat = Mat()
        val morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(inputMat, morphMat, Imgproc.MORPH_OPEN, morphKernel)
        return morphMat
    }
    private fun convertToBitmap(inputMat: Mat): Bitmap {
        val outputBitmap = Bitmap.createBitmap(inputMat.width(), inputMat.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(inputMat, outputBitmap)
        return outputBitmap
    }
}