package com.example.thesisocr

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.thesisocr.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {
    // ONNX Variables
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    // TODO: Integrate PreProcessing.
    // TODO: Move camera call and file picker functions to separate class.
    private lateinit var binding: ActivityMainBinding
    private var imageView: ImageView? = null
    private val preProcessing = PreProcessing()
    private val textRecognition = PaddleRecognition()
    private val textDetection = PaddleDetector()
    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        displayImageFromUri(uri)
        if (uri != null){
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            Log.d("Photo Picker", "Photo selected: $uri")
            debugGetModelInfo(1)
            debugGetModelInfo(2)
            // imagePreProcess(bitmap)
            neuralNetProcess(bitmap)
        } else {
            Log.d("Photo Picker", "No photo selected.")
        }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        // Load OpenCV
        OpenCVLoader.initLocal()
        if(OpenCVLoader.initLocal()){
            Log.e("MyTag","OpenCV Loaded.")
        } else {
            Log.e("MyTag","OpenCV Not Loaded.")
        }
        // ONNX Model Stuff
        // Text Detection Model
        // Android Application Stuff
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val btnCapture = findViewById<Button>(R.id.btnCapture)
        val btnSelectImage = findViewById<Button>(R.id.btnSelectImage)
        imageView = findViewById(R.id.imageView)
        btnSelectImage.setOnClickListener {
            pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
    }
    private fun imagePreProcess(bitmap: Bitmap){
        Log.d("Image Pre-Processing", "Image Pre-Processing Started.")
        val preProcessedBitmap = preProcessing.imagePreProcess(bitmap)
        Log.d("Image Pre-Processing", "Image Pre-Processing Completed.")
        displayImage(preProcessedBitmap)
        Log.d("Output Image", "Output Image Saved to ${Environment.getExternalStorageDirectory().toString() + "/Pictures/output.jpg"}")
    }
    private fun neuralNetProcess(bitmap: Bitmap){
        val rescaledBitmap = rescaleBitmap(bitmap, 640, 480)
        Log.d("Neural Network Processing", "Neural Network Processing Started.")
        // Run detection model.
        var detectionInferenceTime = System.currentTimeMillis()
        var selectedModelByteArray = selectModel(1)
        ortSession = ortEnv.createSession(selectedModelByteArray, OrtSession.SessionOptions())
        val detectionResult = textDetection.detect(rescaledBitmap, ortEnv, ortSession)
        if (detectionResult != null) {
            // Display image to UI.
            // displayImage(result.outputBitmap)
            // Save image to device [DEBUGGING].
            // saveImage(result.outputBitmap, Environment.getExternalStorageDirectory().toString() + "/Pictures/output.jpg")
            // Crop image to bounding boxes.
            val croppedBitmapList = PaddleDetectorPostProcessing().cropBitmapToBoundingBoxes(rescaleBitmap(bitmap,640,480), detectionResult.boundingBoxList)
            detectionInferenceTime = System.currentTimeMillis() - detectionInferenceTime
            Log.d("Text Detection", "Detection (inc. processing) Inference Time: $detectionInferenceTime ms")
            // Run recognition model.
            selectedModelByteArray = selectModel(2)
            ortSession.close()
            ortSession = ortEnv.createSession(selectedModelByteArray, OrtSession.SessionOptions())
            val recognitionResult = textRecognition.recognize(croppedBitmapList, ortEnv, ortSession)
        }
        Log.d("Text Recognition", detectionResult.toString())
        // ortSession = ortEnv.createSession(selectedModelByteArray, OrtSession.SessionOptions())
        Log.d("Neural Network Processing", "Neural Network Processing Completed.")
        Log.d("Output Image", "Output Image Saved to ${Environment.getExternalStorageDirectory().toString() + "/Pictures/output.jpg"}")
    }
    private fun displayImage(bitmap: Bitmap?) {
        imageView!!.visibility = View.VISIBLE
        imageView!!.setImageBitmap(bitmap)
    }
    private fun displayImageFromUri(imageUri: Uri?) {
        imageView!!.visibility = View.VISIBLE
        imageView!!.setImageURI(imageUri)
    }
    private fun rescaleBitmap(bitmap: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, false)
    }
    private fun selectModel(modelNum: Int): ByteArray{
        val modelPackagePath = when (modelNum) {
            1 -> R.raw.det_model
            2 -> R.raw.rec_model
            else -> R.raw.det_model
        }
        return resources.openRawResource(modelPackagePath).readBytes()
    }
    private fun gatherModelOutputInputInfo(modelToLoad: ByteArray){
        ortSession = ortEnv.createSession(modelToLoad, OrtSession.SessionOptions())
        val inputInfo = ortSession.inputInfo
        val outputInfo = ortSession.outputInfo
        Log.d("Model Info", "Input Info: $inputInfo")
        Log.d("Model Info", "Output Info: $outputInfo")
    }
    private fun debugGetModelInfo(modelSelect: Int){
        val selectedModelByteArray = selectModel(modelSelect)
        gatherModelOutputInputInfo(selectedModelByteArray)
    }
    private fun saveImage(bitmap: Bitmap, filename: String){
        val fileOutputStream = FileOutputStream(filename)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
        fileOutputStream.close()
    }
}