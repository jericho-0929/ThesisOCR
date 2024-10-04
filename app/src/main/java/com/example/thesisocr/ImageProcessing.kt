package com.example.thesisocr

import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.core.graphics.blue
import androidx.core.graphics.green
import androidx.core.graphics.red
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
        Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_RGBA2RGB)
        return inputMat
    }
    fun adaptBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        // Check if inputBitmap is in 4:3 aspect ratio.
        if (bitmap.width % 4 == 0 && bitmap.height % 3 == 0) {
            return rescaleBitmap(bitmap, targetWidth, targetHeight)
        } else {
            // Rescale the inputBitmap until it fits the target resolution.
            fun recursiveRescaleBitmap(inputBitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
                val rescaledBitmap = rescaleBitmapOpenCV(inputBitmap, inputBitmap.width / 1.125, inputBitmap.height / 1.125)
                return if (rescaledBitmap.width <= targetWidth && rescaledBitmap.height <= targetHeight) {
                    rescaledBitmap
                } else {
                    recursiveRescaleBitmap(rescaledBitmap, targetWidth, targetHeight)
                }
            }
            val rescaledBitmap = recursiveRescaleBitmap(bitmap, targetWidth, targetHeight)
            // Put the rescaledBitmap on a 4:3 aspect ratio canvas.
            val canvasBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(canvasBitmap)
            canvas.drawBitmap(rescaledBitmap, 0.0f, 0.0f, null)
            return canvasBitmap
        }
    }
    // Detection pre-processing & post-processing functions.
    fun processDetectionOutputMask(inputBitmap: Bitmap): Bitmap {
        // Convert the bitmap to a Mat.
        val inputMat = convertBitmapToMat(inputBitmap)
        // val erosionMat = erosion(inputMat, 5.0, 5.0)
        return convertToBitmap(dilation(inputMat, 3.0, 3.0))
    }
    fun processImageForDetection(inputBitmap: Bitmap): Bitmap {
        // val thresholdMat = thresholdBitmap(inputBitmap)
        // val thresholdMat = imageThresholding(convertToGrayscaleMat(inputBitmap))
        val blurredMat = imageBlur(convertBitmapToMat(inputBitmap))
        return convertToBitmap(
                blurredMat
        )
    }
    private fun invertImage(inputBitmap: Bitmap): Bitmap {
        val inputMat = convertBitmapToMat(inputBitmap)
        Core.bitwise_not(inputMat, inputMat)
        return convertToBitmap(inputMat)
    }
    // Recognition pre-processing functions.
    fun processImageForRecognition(inputBitmap: Bitmap): Bitmap {
        val thresholdMat = imageThresholding(convertToGrayscaleMat(inputBitmap))
        // Blur twice.
        val blurredMat = imageBlur(thresholdMat)
        val openedMat = opening(blurredMat, 3.0, 3.0)
        return convertToBitmap(imageBlur(openedMat))
    }
    fun convertToGrayscaleMat(inputBitmap: Bitmap): Mat {
        val inputMat = Mat()
        Utils.bitmapToMat(inputBitmap, inputMat)
        val grayMat = Mat()
        Imgproc.cvtColor(inputMat, grayMat, Imgproc.COLOR_BGR2GRAY)
        return grayMat
    }
    private fun thresholdBitmap(inputBitmap: Bitmap): Mat {
        val floatArray = convertImageToFloatArray(inputBitmap)
        val normalizedArray = normalizeFloatArray(floatArray)
        val bitmap = convertFloatArrayToImage(normalizedArray)
        return convertBitmapToMat(invertImage(bitmap))
    }
    private fun imageBlur(inputMat: Mat): Mat {
        val blurredMat = Mat()
        Imgproc.bilateralFilter(inputMat, blurredMat, 5, 75.0, 75.0)
        return blurredMat
    }
    private fun imageThresholding(inputMat: Mat): Mat {
        val thresholdMat = Mat()
        // Imgproc.threshold(inputMat, thresholdMat, 165.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        Imgproc.adaptiveThreshold(inputMat, thresholdMat, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 95, 47.0)
        return thresholdMat
    }
    private fun imageSharpening(inputMat: Mat): Mat {
        val sharpenedMat = Mat()
        val kernel = Mat(3, 3, CvType.CV_32F)
        kernel.put(0, 0, 0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0)
        Imgproc.filter2D(inputMat, sharpenedMat, -1, kernel)
        return sharpenedMat
    }
    fun dilation(inputMat: Mat, x: Double, y: Double): Mat {
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
    fun erosion(inputMat: Mat, x: Double, y: Double): Mat {
        val erodedMat = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(x, y))
        Imgproc.erode(inputMat, erodedMat, kernel)
        return erodedMat
    }
    fun contourFiltering() {
        // TODO: Implement contour filtering.
    }
    fun applyMask(inputBitmap: Bitmap, resources: Resources): Bitmap {
        // Use philsys_mask.jpg as the mask.
        val maskBitmap = rescaleBitmap(
            BitmapFactory.decodeResource(resources, R.drawable.id_mask),
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
    // TODO: Remove functions from other classes.
    fun normalizeFloatArray(inputArray: Array<Array<Array<FloatArray>>>): Array<Array<Array<FloatArray>>> {
        // Normalize the inputArray through the use of mean and standard deviation.
        // Dimensions: (Batch Size, Channels, Width, Height)
        val scale = 1/255.0f
        val width = inputArray[0][0].size
        val height = inputArray[0][0][0].size
        val normalizedArray = Array(1) { Array(3) { Array(width) { FloatArray(height) } } }
        // Mean and standard deviation values for each channel.
        val meanValues = floatArrayOf(0.615f,0.667f,0.724f)
        val stdValues = floatArrayOf(0.232f,0.240f,0.240f)
        for (k in 0 until 3) {
            for (i in 0 until width) {
                for (j in 0 until height) {
                    normalizedArray[0][k][i][j] = (inputArray[0][k][i][j] * scale - meanValues[k]) / stdValues[k]
                }
            }
        }
        return normalizedArray
    }
    fun convertFloatArrayToImage(floatArray: Array<Array<Array<FloatArray>>>): Bitmap {
        val width = floatArray[0][0].size
        val height = floatArray[0][0][0].size
        val channels = floatArray[0].size
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        var minVal = 0
        var maxVal = 0
        for (i in 0 until width) {
            for (j in 0 until height) {
                for (k in 0 until channels) {
                    minVal = minOf(minVal, floatArray[0][k][i][j].toInt())
                    maxVal = maxOf(maxVal, floatArray[0][k][i][j].toInt())
                }
            }
        }
        for (i in 0 until width) {
            for (j in 0 until height) {
                val red = ((floatArray[0][0][i][j] - minVal) / (maxVal - minVal) * 255).toInt()
                val green = ((floatArray[0][1][i][j] - minVal) / (maxVal - minVal) * 255).toInt()
                val blue = ((floatArray[0][2][i][j] - minVal) / (maxVal - minVal) * 255).toInt()
                val pixel = 0xff shl 24 or (red shl 16) or (green shl 8) or blue
                bitmap.setPixel(i, j, pixel)
            }
        }
        return bitmap
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
    private fun rescaleBitmapOpenCV(inputBitmap: Bitmap, newWidth: Double, newHeight: Double): Bitmap {
        val inputMat = convertBitmapToMat(inputBitmap)
        val resizedMat = Mat()
        Imgproc.resize(inputMat, resizedMat, Size(newWidth, newHeight))
        return convertToBitmap(resizedMat)
    }
}