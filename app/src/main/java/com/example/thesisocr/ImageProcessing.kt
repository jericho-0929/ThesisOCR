package com.example.thesisocr

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class ImageProcessing {
    // General image processing functions.
    fun rescaleBitmap(inputBitmap: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(inputBitmap, newWidth, newHeight, false)
    }
    private fun convertBitmapToMat(inputBitmap: Bitmap): Mat {
        val inputMat = Mat()
        Utils.bitmapToMat(inputBitmap, inputMat)
        Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_RGBA2BGR)
        return inputMat
    }
    // Detection pre-processing functions.
    // Blacken out 25% of the image's top and 10% of the image's right sections.
    fun processImageForDetection(inputBitmap: Bitmap): Bitmap {
        val blurredMat = imageBlur(
            sectionRemoval(
                convertBitmapToMat(inputBitmap)
            )
        )
        val dilatedMat = dilation(blurredMat)
        val openedMat = opening(dilatedMat)
        return convertToBitmap(openedMat)
    }
    private fun sectionRemoval(inputMat: Mat): Mat {
        // Channel count is 3.
        val width = inputMat.width()
        val height = inputMat.height()
        val topSectionHeight = (height * 0.25).toInt()
        val rightSectionWidth = (width * 0.10).toInt()
        // Whiten out the topmost section.
        for (i in 0 until topSectionHeight) {
            for (j in 0 until width) {
                inputMat.put(i, j, 255.0, 255.0, 255.0)
            }
        }
        // Whiten out the rightmost section.
        for (i in 0 until height) {
            for (j in 0 until rightSectionWidth) {
                inputMat.put(i, width - j, 255.0, 255.0, 255.0)
            }
        }
        return inputMat
    }
    // Recognition pre-processing functions.
    fun processImageForRecognition(inputBitmap: Bitmap): Bitmap {
        val grayMat = convertToGrayscaleMat(inputBitmap)
        val blurredMat = imageBlur(grayMat)
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
    private fun imageBlur(inputMat: Mat): Mat {
        val blurredMat = Mat()
        Imgproc.bilateralFilter(inputMat, blurredMat, 5, 75.0, 75.0)
        return blurredMat
    }
    private fun imageThresholding(inputMat: Mat): Mat {
        val thresholdMat = Mat()
        Imgproc.threshold(inputMat, thresholdMat, 165.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        return thresholdMat
    }
    private fun imageSharpening(inputMat: Mat): Mat {
        val sharpenedMat = Mat()
        val kernel = Mat(3, 3, CvType.CV_32F)
        kernel.put(0, 0, 0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0)
        Imgproc.filter2D(inputMat, sharpenedMat, -1, kernel)
        return sharpenedMat
    }
    private fun dilation(inputMat: Mat): Mat {
        val dilatedMat = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.dilate(inputMat, dilatedMat, kernel)
        return dilatedMat
    }
    private fun opening(inputMat: Mat): Mat {
        val openedMat = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.morphologyEx(inputMat, openedMat, Imgproc.MORPH_OPEN, kernel)
        return openedMat
    }
    private fun contourFiltering() {
        // TODO: Implement contour filtering.
    }
    private fun convertToBitmap(inputMat: Mat): Bitmap {
        val outputBitmap = Bitmap.createBitmap(inputMat.width(), inputMat.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(inputMat, outputBitmap)
        return outputBitmap
    }
}