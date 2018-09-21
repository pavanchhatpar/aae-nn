import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class PredictBE {
    public static void main(String[] args) throws Exception {
        Session session = SavedModelBundle.load("../../BE-project/tf/meta/nn_v1-1-geohash_export/1520260725", "serve").session();
        //all prediction samples reshaped as 1-d array
        String geohashx = "te7ue69y8405";
        byte[][] matrixgeo = new byte[1][];
        matrixgeo[0] = geohashx.getBytes("UTF-8");
        float weekdayx[] = new float[] {2f};
        float hoursx[] = new float[] {21f};
        float minutesx[] = new float[] {2f};
        int nSamples = 1;
        int nFeatures = 1;
        Tensor<String> geohash =
            Tensor.create(matrixgeo, String.class);
	Tensor weekday =
            Tensor.create(
                new long[] {nSamples, nFeatures},
                FloatBuffer.wrap(weekdayx));
	Tensor hours =
            Tensor.create(
                new long[] {nSamples, nFeatures},
                FloatBuffer.wrap(hoursx));
	Tensor minutes =
            Tensor.create(
                new long[] {nSamples, nFeatures},
                FloatBuffer.wrap(minutesx));

        // You can inspect them on the command-line with saved_model_cli:
        // $ saved_model_cli show --dir $EXPORT_DIR --tag_set serve --signature_def predict
        final String geohashName = "Placeholder:0";
        final String weekdayName = "Placeholder_1:0";
        final String hourName = "Placeholder_2:0";
        final String minName = "Placeholder_3:0";
        final String className = "dnn/head/predictions/ExpandDims:0";
        
        List<Tensor<?>> outputs = session.runner()
            .feed(geohashName, geohash)
            .feed(weekdayName, weekday)
            .feed(hourName, hours)
            .feed(minName, minutes)
            .fetch(className)
            .run();
        // System.out.println(outputs.get(0).toString()); // use this to find dimension of tensor
        // Outer dimension is batch size; inner dimension is number of classes
        // INT64 long; INT32 int; DOUBLE double; FLOAT double; STRING byte
        long[][] scores = new long[1][1];
        outputs.get(0).copyTo(scores);
        System.out.println(Arrays.deepToString(scores));
    }
}
