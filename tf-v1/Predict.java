import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class Predict {
    public static void main(String[] args) {
        Session session = SavedModelBundle.load("tf/psychometric_trained/1514380668", "serve").session();
        //all prediction samples reshaped as 1-d array
        double dx[] = new double[] {
            4, 0, 0, 0, 6, 0, 0, 0, 0.362266659736633, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.120166666805744, 0.188624992966652, 0.181966662406921, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.785141661763191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.225933328270912, 0, 0, 0, 0, 0.212066665291786, 0.146899998188019, 0, 0.511299997568131, 0, 0, 0.197500005364418, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.432599991559982, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.181700006127357, 0.182333335280418, 0, 0, 0.143299996852875, 0.189533337950706, 0, 0, 0, 0, 0, 0.151233330368996, 0, 0.566366672515869, 0, 0, 0, 0, 0.15763333439827, 0, 0, 0,0, 0, 0, 0.240799993276596, 0, 0, 0, 0, 0.124866664409637, 0, 0, 0, 0, 0.169599995017052, 0, 0, 0.224333330988884, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18529999256134, 7, 0, 3, 0, 0, 0, 0, 0
        };
        float[] fx = new float[dx.length];
        for (int i = 0 ; i < dx.length; i++)
        {
            fx[i] = (float) dx[i];
        }
        int nSamples = 1;
        int nFeatures = 142;
        Tensor x =
            Tensor.create(
                new long[] {nSamples, nFeatures},
                FloatBuffer.wrap(fx));

        // You can inspect them on the command-line with saved_model_cli:
        // $ saved_model_cli show --dir $EXPORT_DIR --tag_set serve --signature_def predict
        final String xName = "Placeholder:0";
        final String className = "dnn/head/logits:0";
        
        List<Tensor<?>> outputs = session.runner()
            .feed(xName, x)
            .fetch(className)
            .run();
        // System.out.println(outputs.get(0).toString()); // use this to find dimension of tensor
        // Outer dimension is batch size; inner dimension is number of classes
        // INT64 long; INT32 int; DOUBLE double; FLOAT double; STRING byte
        float[][] scores = new float[1][46];
        outputs.get(0).copyTo(scores);
        System.out.println(Arrays.deepToString(scores));
    }
}