package com.example.thesisocr

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.icu.text.SimpleDateFormat
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageCapture
import androidx.core.content.ContextCompat
import com.example.thesisocr.databinding.ActivityCameraBinding
import org.opencv.android.OpenCVLoader
import java.io.FileOutputStream
import java.io.OutputStream
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import com.google.gson.Gson
import java.io.File

class MainActivity : AppCompatActivity() {
    // Model Vocabulary from en_dict.txt raw resource file.
    private lateinit var modelVocab: List<String>
    // ONNX Variables
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    // Camera Variables
    private lateinit var viewBinding: ActivityCameraBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraActivity: CameraActivity
    // UI Variables.
    private var imageView: ImageView? = null
    private lateinit var textView: TextView
    private lateinit var inferenceView: TextView
    // Everything else.
    private lateinit var modelProcessing: ModelProcessing
    private lateinit var modelResults: ModelProcessing.ModelResults

    override fun onCreate(savedInstanceState: Bundle?) {
        // Load OpenCV
        OpenCVLoader.initLocal()
        if(OpenCVLoader.initLocal()){
            Log.e("MyTag","OpenCV Loaded.")
        } else {
            Log.e("MyTag","OpenCV Not Loaded.")
        }
        // Initialize Model Processing and Camera Activity
        cameraActivity = CameraActivity()
        modelProcessing = ModelProcessing(resources)
        // Warm-up
        Toast.makeText(baseContext,
            "Initializing models.",
            Toast.LENGTH_SHORT).show()
        modelProcessing.warmupThreads()
        // Model Info
        modelProcessing.debugGetModelInfo(1)
        modelProcessing.debugGetModelInfo(2)
        // Android Application Stuff
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        viewBinding = ActivityCameraBinding.inflate(layoutInflater)
        val textView = findViewById<TextView>(R.id.textView)
        textView.movementMethod = android.text.method.ScrollingMovementMethod()
        // Button declarations
        val btnCallCamera = findViewById<Button>(R.id.btnCallCamera)
        val btnSelectImage = findViewById<Button>(R.id.btnSelectImage)
        val btnDebugSave = findViewById<Button>(R.id.debugSave)
        imageView = findViewById(R.id.imageView)

        // Button Listeners
        // Select Image Button
        btnSelectImage.setOnClickListener {
            pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        // Call Camera Button
        btnCallCamera.setOnClickListener {
            startCameraActivity.launch(Intent(this, CameraActivity::class.java))
        }
        // Debug Save Button
        btnDebugSave.setOnClickListener {
            debugSaveImages()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }
    // Variables with lazy initialization
    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        displayImageFromUri(uri)
        if (uri != null){
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            // Process the image.
            processBitmap(bitmap)
        } else {
            Log.d("Photo Picker", "No photo selected.")
        }
    }
    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()){
        permissions ->
        var permissionGranted = true
        permissions.entries.forEach {
            if (it.key in REQUIRED_PERMISSIONS && !it.value)
                permissionGranted = false
        }
        if (!permissionGranted) {
            Toast.makeText(baseContext,
                "Permission request denied",
                Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(baseContext,
                "Camera open.",
                Toast.LENGTH_SHORT).show()
        }
    }
    private val startCameraActivity = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val data: Intent? = result.data
            val bitmapUri = Uri.parse(data?.getStringExtra("data"))
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, bitmapUri)
            // Process the image.
            processBitmap(bitmap)
        }
    }
    // Functions
    private fun processBitmap(bitmap: Bitmap) {
        // Process the image.
        modelResults = modelProcessing.processImage(bitmap)
        // Check if recognition result is null.
        if (modelResults.recognitionResult == null){
            Toast.makeText(baseContext,
                "Image not good. Please try again.",
                Toast.LENGTH_SHORT).show()
            return
        } else {
            // Display the image and recognition results.
            displayImage(modelResults.detectionResult.outputBitmap)
            displayRecognitionResults(modelResults.recognitionResult!!.listOfStrings)
            // Display inference times to the user.
            Toast.makeText(
                baseContext,
                "Detection Inference Time: ${modelResults.detectionResult.inferenceTime.inWholeMilliseconds.toInt()} ms",
                Toast.LENGTH_LONG
            ).show()
            Toast.makeText(
                baseContext,
                "Recognition Inference Time: ${modelResults.recognitionResult!!.inferenceTime.inWholeMilliseconds.toInt()} ms",
                Toast.LENGTH_LONG
            ).show()
            displayInferenceTime(
                modelResults.detectionResult.inferenceTime.inWholeMilliseconds,
                modelResults.recognitionResult!!.inferenceTime.inWholeMilliseconds
            )
        }
    }
    private fun displayImage(bitmap: Bitmap?) {
        imageView!!.visibility = View.VISIBLE
        imageView!!.setImageBitmap(bitmap)
    }
    private fun displayImageFromUri(imageUri: Uri?) {
        imageView!!.visibility = View.VISIBLE
        imageView!!.setImageURI(imageUri)
    }
    private fun displayRecognitionResults(listOfStrings: MutableList<String>) {
        textView = findViewById(R.id.textView)
        textView.visibility = View.VISIBLE
        textView.text = "Recognition Results:"
        for (string in listOfStrings) {
            textView.append("\n")
            textView.append(string)
        }
    }
    private fun displayInferenceTime(detectionInferenceTime: Long, recognitionInferenceTime: Long) {
        inferenceView = findViewById(R.id.inferenceView)
        inferenceView.visibility = View.VISIBLE
        inferenceView.text = "Detection inference time: $detectionInferenceTime ms"
        inferenceView.append("\nRecognition inference time: $recognitionInferenceTime ms")
    }
    // Debug Functions
    private fun debugSaveImages(){
        // Grab bitmap contents of modelProcessing.
        val detectionBitmapMask = modelResults.detectionResult.outputMask
        val detectionBitmap = modelResults.detectionResult.outputBitmap
        val recognitionInputBitmapList = modelResults.recogInputBitmapList
        val dateFormat = SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS", Locale.US)
        // Save the images.
        val yeetLocation = "/storage/emulated/0/Documents/ThesisOCR"
        val detectionMaskBool = saveImage(detectionBitmapMask, "$yeetLocation/detection_mask" + dateFormat.format(System.currentTimeMillis()) + ".jpg")
        val detectionOutputBool = saveImage(detectionBitmap, "$yeetLocation/detection_result" + dateFormat.format(System.currentTimeMillis()) + ".jpg")
        /*for (i in 0 until recognitionInputBitmapList.size){
            val recognitionInputBool = saveImage(recognitionInputBitmapList[i], "$yeetLocation/recognition_input_$i.jpg")
        }*/
        if (detectionMaskBool && detectionOutputBool){
            Toast.makeText(baseContext,
                "Images saved to $yeetLocation.",
                Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(baseContext,
                "Image saving failed.",
                Toast.LENGTH_SHORT).show()
        }
        val isRecognitionOutputSaved = saveStringListAsJson(modelResults.recognitionResult!!.listOfStrings, "$yeetLocation/recognition_output" + dateFormat.format(System.currentTimeMillis()) + ".json")
        val isDetectionOutputSaved = saveBoundingBoxCoordinatesAsJson(modelResults.detectionResult.boundingBoxList, "$yeetLocation/detection_output" + dateFormat.format(System.currentTimeMillis()) + ".json")
        if (isRecognitionOutputSaved && isDetectionOutputSaved){
            Toast.makeText(baseContext,
                "Text and Bounding Box Coordinates saved to $yeetLocation.",
                Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(baseContext,
                "Text and Bounding Box Coordinates saving failed.",
                Toast.LENGTH_SHORT).show()
        }
    }
    private fun saveImage(bitmap: Bitmap, filename: String): Boolean {
        val fileOutputStream = FileOutputStream(filename)
        val saveBool = bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
        fileOutputStream.close()
        return saveBool
    }
    private fun saveStringListAsJson(inputList: List<String>, filename: String): Boolean {
        val gson = Gson()
        val jsonString = gson.toJson(inputList)
        File(filename).writeText(jsonString)
        return true
    }
    private fun saveBoundingBoxCoordinatesAsJson(inputList: List<PaddleDetector.BoundingBox>, filename: String): Boolean {
        val gson = Gson()
        val jsonString = gson.toJson(inputList)
        File(filename).writeText(jsonString)
        return true
    }
    private fun OutputStream.saveStringListAsCSV(context: Context, inputList: List<String>): Boolean {
        // Write inputList to specified filename.
        val writer = bufferedWriter()
        inputList.forEach {
            writer.write(it)
            writer.newLine()
        }
        writer.flush()
        writer.close()
        return true
    }
    private fun OutputStream.saveBoundingBoxCoordinatesAsCSV(context: Context, inputList: List<PaddleDetector.BoundingBox>): Boolean {
        // Each row has following entries: x, y, width, height.
        val writer = bufferedWriter()
        inputList.forEach{
            writer.write("${it.x}, ${it.y}, ${it.width}, ${it.height}")
            writer.newLine()
        }
        writer.flush()
        writer.close()
        return true
    }
    // CameraX Functions
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
    private fun requestPermissions() {
        requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
    }
    override fun onDestroy() {
        super.onDestroy()
        if (this::ortSession.isInitialized) {
            ortSession.close()
        }
        ortEnv.close()
        cameraExecutor.shutdown()
    }
    companion object {
        const val TAG = "CameraXApp"
        const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS = arrayOf(android.Manifest.permission.CAMERA)
    }
}