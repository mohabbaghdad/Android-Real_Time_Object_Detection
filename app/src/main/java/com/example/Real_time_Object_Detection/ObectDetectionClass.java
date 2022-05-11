package com.example.Real_time_Object_Detection;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class ObectDetectionClass {
    // used to load model and predicts
    private Interpreter interpreter;
    //used to store label in arrayList
    private List<String> labelList;
    //Input Size
    private int Input_Size;
    //for RGB = 3
    private int Pixel_Size = 3;
    // image Mean
    private int Image_Mean = 0;
    // image Standered Divsion
    private float Image_STD = 255.0f;

    // use to initlize gpu in app
    private GpuDelegate gpuDelegate;

    private int height= 0;
    private int width = 0;

    ObectDetectionClass(int input_Size,AssetManager assetManger, String modelPath,String labelPath)throws IOException{
        this.Input_Size=input_Size;
        // use to define gpu or cpu && number of threads
        Interpreter.Options options = new Interpreter.Options();
        this.gpuDelegate = new GpuDelegate();
        //to use gpu not mobile cpu
        options.addDelegate(gpuDelegate);
        //set threads according to your phone cabability
        options.setNumThreads(4);

        // load model deceleared down in function loadModdelFile
        interpreter = new Interpreter(loadModelFile(assetManger,modelPath),options);

        // load labelmap deceleared down in function loadLabelList
        labelList = loadLabelList(assetManger,labelPath);


    }
    private List<String> loadLabelList(AssetManager assetManger, String labelPath) throws IOException {

        // use to store Label
        List<String> labelList = new ArrayList<>();
        // use to create a new reader
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManger.open(labelPath)));
        String line;

        //loop through each line and store it to LabelList

        while((line = reader.readLine())!=null){
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }
    private ByteBuffer loadModelFile(AssetManager assetManger, String modelPath) throws IOException {

        // use to get description of the file
        AssetFileDescriptor fileDescriptor = assetManger.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
}
