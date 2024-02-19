package com.example.thesisocr

import android.graphics.Bitmap
/**
 * PaddleDetectorPostProcessing class for processing
 * output from the PaddleOCR's text detection model.
 *
 * Functions implemented/to be implemented:
 * Creation of cropped images from extracted bounding boxes.
 */
class PaddleDetectorPostProcessing {
    fun cropBitmapToBoundingBoxes(inputBitmap: Bitmap, boundingBoxList: List<PaddleDetector.BoundingBox>): List<Bitmap> {
        // BoundingBox variables: x, y, width, height
        val croppedBitmapList = mutableListOf<Bitmap>()
        for (boundingBox in boundingBoxList){
            val croppedBitmap = Bitmap.createBitmap(inputBitmap, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height)
            croppedBitmapList.add(croppedBitmap)
        }
        return croppedBitmapList
    }
    fun resizeToHeight48Pixels(bitmap: Bitmap): Bitmap {
        val aspectRatio = bitmap.width.toFloat() / bitmap.height.toFloat()
        val newWidth = (48 * aspectRatio).toInt()
        return Bitmap.createScaledBitmap(bitmap, newWidth, 48, false)
    }
}