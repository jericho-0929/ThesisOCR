package com.example.thesisocr

import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
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
        Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_RGBA2BGR)
        return inputMat
    }
    // Detection pre-processing functions.
    fun processImageForDetection(inputBitmap: Bitmap): Bitmap {
        val blurredMat = imageBlur(
            sectionRemoval(
                convertToGrayscaleMat(inputBitmap)
            )
        )
        val sharpenedMat = imageSharpening(imageBlur(blurredMat))
        val openedMat = opening(sharpenedMat)
        return convertToBitmap(openedMat)
    }
    // Blacken out a percentage of the image's top and of the image's right.
    private fun sectionRemoval(inputMat: Mat): Mat {
        // Channel count is 3.
        val width = inputMat.width()
        val height = inputMat.height()
        val topSectionHeight = (height * 0.30).toInt()
        val rightSectionWidth = (width * 0.10).toInt()
        // Whiten out the topmost section.
        for (i in 0 until topSectionHeight) {
            for (j in 0 until width) {
                inputMat.put(i, j, 255.0, 255.0, 255.0)
            }
        }
        // Whiten out the rightmost section
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
        // Blur twice.
        val blurredMat = imageBlur(imageBlur(grayMat))
        val thresholdMat = imageThresholding(blurredMat)
        val openedMat = opening(thresholdMat)
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
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
        Imgproc.dilate(inputMat, dilatedMat, kernel)
        return dilatedMat
    }
    fun opening(inputMat: Mat): Mat {
        val openedMat = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
        Imgproc.morphologyEx(inputMat, openedMat, Imgproc.MORPH_OPEN, kernel)
        return openedMat
    }
    fun contourFiltering() {
        // TODO: Implement contour filtering.
    }
    fun applyMask(inputBitmap: Bitmap, resources: Resources): Bitmap {
        // Use philsys_mask.jpg as the mask.
        val maskBitmap = rescaleBitmap(
            BitmapFactory.decodeResource(resources, R.drawable.philsys_mask),
            inputBitmap.width,
            inputBitmap.height
        )
        // Convert bitmaps to scalars.
        val inputMat = convertBitmapToMat(inputBitmap)
        val maskMat = convertBitmapToMat(maskBitmap)
        val outputMat = Mat()
        // Apply bitwise_and operation.
        Core.bitwise_and(inputMat, maskMat, outputMat)
        return convertToBitmap(outputMat)
    }
    fun convertToBitmap(inputMat: Mat): Bitmap {
        val outputBitmap = Bitmap.createBitmap(inputMat.width(), inputMat.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(inputMat, outputBitmap)
        return outputBitmap
    }
}