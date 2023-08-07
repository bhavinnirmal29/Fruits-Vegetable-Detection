package com.example.objectdetection;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.objectdetection.ml.Fv11;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.model.*;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class MainActivity extends AppCompatActivity {

    Button camera_btn, analyze_btn;
    TextView result;
    int imageSize = 224;
    Bitmap photo;
    String output = "";
    ImageView captured_Image;
    private static final int IMAGE_CAPTURE_CODE = 1001;
    private static final int PERMISSION_CODE = 1000;
    private static final int pic_id = 123;
    private static final int CAMERA_REQUEST = 1888;
    private Interpreter tflite;
    Bitmap bmp;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        captured_Image = (ImageView) findViewById(R.id.image_view);
        camera_btn = findViewById(R.id.btn1);
        analyze_btn = findViewById(R.id.analyze_btn);
        result = findViewById(R.id.result_text);
        camera_btn.setOnClickListener(view -> {
            Intent camera_intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            // Start the activity with camera_intent, and request pic id
            startActivityForResult(camera_intent, CAMERA_REQUEST);
        });

        analyze_btn.setOnClickListener(view -> {
            output = classifyImage(bmp);

            result.setText(output);
        });

    }


//Loading the tflite model
    private MappedByteBuffer loadModel() throws IOException {
        // Load the TensorFlow Lite model from the assets folder
        AssetFileDescriptor fileDescriptor = getAssets().openFd("FV1 (1).tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
//        ByteBuffer modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
//
//        // Create an instance of the Interpreter class to run inference with the model
//        Interpreter.Options options = new Interpreter.Options();
//        tflite = new Interpreter(modelBuffer, options);
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }

    // Preprocess the image
//    private Bitmap preprocessImage(Bitmap inputBitmap) {
//        // Create an instance of the ImageProcessor class from the TensorFlow Lite Support Library
//        ImageProcessor imageProcessor = new ImageProcessor.Builder()
//                .add(new ResizeOp(inputBitmap.getWidth(), inputBitmap.getHeight(), ResizeOp.ResizeMethod.BILINEAR))
//                .add(new NormalizeOp(0.0f, 255.0f))
//                .build();
//
//        // Preprocess the input bitmap using the ImageProcessor
//        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
//        tensorImage.load(inputBitmap);
//        tensorImage = imageProcessor.process(tensorImage);
//        return tensorImage.getBitmap();
//    }


//    private float[] runInference(Bitmap inputBitmap) {
//        // Preprocess the input bitmap
//        Bitmap preprocessedBitmap = preprocessImage(inputBitmap);
//
//        // Create a ByteBuffer to hold the input image data
//        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(1 * preprocessedBitmap.getByteCount());
//        inputBuffer.order(ByteOrder.nativeOrder());
//        inputBuffer.rewind();
//
//        // Copy the pixel values of the preprocessed bitmap into the input buffer
//        preprocessedBitmap.copyPixelsToBuffer(inputBuffer);
//
//        // Run inference with the model
////        float[][] output = new float[1][NUM_CLASSES];
//        tflite.run(inputBuffer, output);
//
//        // Get the prediction result from the output
//        return output[0];
//    }
        public String classifyImage(Bitmap image) {
            String result1 = "";
            try {
                Fv11 model = Fv11.newInstance(getApplicationContext());

                // Creates inputs for reference.
                TensorImage image1 = TensorImage.fromBitmap(image);

                // Runs model inference and gets result.
                Fv11.Outputs outputs = model.process(image1);
                List<Category> probability = outputs.getProbabilityAsCategoryList();
                System.out.println(probability);

                // Convert the list of categories to a comma-separated string
                // Find the category with the highest score
                Category mostLikelyCategory = null;
                float highestScore = 0.0f;
                for (Category category : probability) {
                    if (category.getScore() > highestScore) {
                        mostLikelyCategory = category;
                        highestScore = category.getScore();
                    }
                }

                // Print the most likely category and its score
                result1 = "Result: "+ mostLikelyCategory.getLabel().toUpperCase()+ " (" + highestScore + ")";
                //
                System.out.println(result1);

                // Releases model resources if no longer used.
                model.close();

            } catch (IOException e) {
                // TODO Handle the exception
            }
            return result1;
        }
        protected void onActivityResult(int requestCode, int resultCode, Intent data) {
            super.onActivityResult(requestCode, resultCode, data);
            if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK) {
                photo = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(photo.getWidth(),photo.getHeight());
                photo = ThumbnailUtils.extractThumbnail(photo,dimension,dimension);
                captured_Image.setImageBitmap(photo);

                photo = Bitmap.createScaledBitmap(photo,imageSize,imageSize,false);

                bmp= photo.copy(Bitmap.Config.ARGB_8888,true) ;
            }
        }

}