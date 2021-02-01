import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class ImageViewer{
    private final JFrame frame;
    private JLabel label;

    public ImageViewer(String s, boolean visible) {
        frame = new JFrame(s);
        label = new JLabel();
        frame.add(label);
        frame.setResizable(true);
        setVisible(visible);
    }

    public void showImage(String filePath, int width, int height) {
        frame.remove(label);
        ImageIcon image = new ImageIcon(filePath);
        label = new JLabel(
                new ImageIcon(image.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT))
        );
        frame.add(label);
        frame.pack();
        frame.validate();
        frame.repaint();
    }

    public void showImage(ImageIcon image, int width, int height) {
        frame.remove(label);
        label = new JLabel(
                new ImageIcon(image.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT))
        );
        frame.add(label);
        frame.pack();
        frame.validate();
        frame.repaint();
    }

    public void showImage(String filePath) {
        showImage(new ImageIcon(filePath));
    }

    public void showImage(ImageIcon image) {
        showImage(image, image.getIconWidth(), image.getIconHeight());
    }

    public void showImage(String filePath, double scale) {
        showImage(new ImageIcon(filePath), scale);
    }

    public void showImage(ImageIcon image, double scale) {
        showImage(image, (int) (image.getIconWidth() * scale), (int) (image.getIconHeight() * scale));
    }
    public void setVisible(boolean visible){
        frame.setVisible(visible);
        frame.repaint();
    }
}
