package com.example.thesisocr

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.core.content.ContextCompat
import com.example.thesisocr.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import java.io.FileOutputStream
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    // Model Vocabulary from en_dict.txt raw resource file.
    private lateinit var modelVocab: List<String>
    // ONNX Variables
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    // TODO: Integrate PreProcessing.
    // TODO: Move camera call and file picker functions to separate class.
    private lateinit var binding: ActivityMainBinding
    private var imageView: ImageView? = null
    private lateinit var modelProcessing: ModelProcessing
    private val imageCapture = ImageCapture.Builder().setCaptureMode(CAPTURE_MODE_MAXIMIZE_QUALITY).build()
    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        displayImageFromUri(uri)
        if (uri != null){
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            Log.d("Photo Picker", "Photo selected: $uri")
            modelProcessing.processImage(bitmap)
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
        // Initialize Model Processing
        modelProcessing = ModelProcessing(resources)
        // Warm-up
        modelProcessing.warmupThreads()
        // Model Info
        modelProcessing.getModelInfo(1)
        modelProcessing.getModelInfo(2)
        // Android Application Stuff
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val btnCapture = findViewById<Button>(R.id.btnCapture)
        val btnSelectImage = findViewById<Button>(R.id.btnSelectImage)
        imageView = findViewById(R.id.imageView)
        btnSelectImage.setOnClickListener {
            pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        btnCapture.setOnClickListener {
            // Set variable for capture output.
            imageCapture.takePicture(ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageCapturedCallback() {
                 override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = imageProxyToBitmap(image)
                     displayImage(bitmap)
                     modelProcessing.processImage(bitmap)
                     super.onCaptureSuccess(image)
                }
                override fun onError(exception: ImageCaptureException) {
                    super.onError(exception)
                }
            })
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
    private fun saveImage(bitmap: Bitmap, filename: String){
        val fileOutputStream = FileOutputStream(filename)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
        fileOutputStream.close()
    }
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer: ByteBuffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
}