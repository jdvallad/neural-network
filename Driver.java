import java.awt.image.BufferedImage;
public class Driver {
    public static void main(String[] args) throws Exception {
        ImageViewer one = new ImageViewer("one");
        ImageViewer two = new ImageViewer("two");
        one.setVisible(true);
        two.setVisible(true);
        BufferedImage image = ImageViewer.pathToImage("./color.jpg");
        int[][][] nums = ImageViewer.imageToList(image, true);
        one.show(image);
        two.show(ImageViewer.listToImage(nums));
        return;
    }
}