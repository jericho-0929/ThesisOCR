package com.example.thesisocr

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.time.temporal.ValueRange

class PreProcessing {
    // DONE: Implement Canny Edge Detection
    // DONE: Implement Hough Transform
    // TODO: Implement class and function calls on MainActivity.kt

    /*
     Call the functions in order of: cannyEdge -> houghTransform().
     Use any available functions to convert resulting Mat() to Bitmap
     for use within Android UI framework.
     */

    companion object {
        init {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
        }
    }
    // Canny Edge Detection
    fun cannyEdge(inputBitmap: Bitmap): Mat {
        // Function Variables
        val inputImage = Mat()
        val grayImage = Mat()
        val noiseReduced = Mat()
        val edges = Mat()
        // Noise reduction kernel sizes
        val noiseKernelX = 3.0
        val noiseKernelY = 3.0
        // Canny parameters
        val thresholdOne = 50.0
        val thresholdTwo = 150.0
        try {
            // Convert Bitmap input into Mat
            Utils.bitmapToMat(inputBitmap, inputImage)
            // Convert to grayscale
            Imgproc.cvtColor(inputImage, grayImage, Imgproc.COLOR_BGR2GRAY)
            // Reduce noise
            // TODO: Test with different methods
            Imgproc.blur(grayImage, noiseReduced, Size(noiseKernelX, noiseKernelY))
            // Run OpenCV Canny
            Imgproc.Canny(noiseReduced, edges, thresholdOne, thresholdTwo)
            // Release Mat objects
            inputImage.release()
            grayImage.release()
            noiseReduced.release()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return edges
    }
    // Standard Hough Line Transform
    fun houghTransform(inputMat: Mat) {
        val houghLines = Mat() // Output variable
        val rho = 1.0
        val theta = Math.PI/180
        val threshold = 150
        Imgproc.HoughLines(inputMat, houghLines, rho, theta, threshold)

    }
}