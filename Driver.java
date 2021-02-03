import java.util.Map;

public class Driver {
    public static void main(String[] args) throws Exception {
        DataIterator nums = new mnistIterator(100, 420);
        nn model = new nn(28 * 28, "logLoss", "./Data/mnistDigitCategorize.ser");
        model.add("sigmoid", 16);
        model.add("leakyRelu", 16);
        model.add("sigmoid", 10);
        //   model.build();
        model = nn.load("./Data/mnistDigitCategorize.ser");
        Map<String, Object> input = Map.of("width", 28, "height", 28, "color", false, "scale", 10.);
        Map<String, Object> output = Map.of("width", 10, "height", 1, "color", false, "scale", 80.);
        model.train(nums, 10, .5, false);
        model.getAccuracy(nums);
    }
}