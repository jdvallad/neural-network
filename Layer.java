 import java.util.*;
 import java.io.*;
 public class Layer {
     double[][] weights;
     double[] biases;
     String activation;
     boolean locked;

     public Layer(double[][] weights, double[] biases, String activation, boolean locked){
         this.locked = locked;
         for(int r = 0; r < weights.length; r++)
             for(int c = 0; c < weights[r].length; c++)
                 this.weights[r][c] = weights[r][c];
         for(int r = 0; r < biases.length; r++)
             this.biases[r] = biases[r];
         this.activation = activation;
     }

     public void lock(){
         locked = true;
         return;
     }

     public void unlock(){
         locked = false;
         return;
     }
     public static nn join(List<Layer> layers,String cost, String saveFile){
         layers = new ArrayList<>(layers);
         Layer temp = layers.remove(0);
         nn res = new nn(temp.biases.length, cost, saveFile);
         while(layers.size() > 0){
             temp = layers.remove(0);
             res.add(temp.activation, temp.biases.length);
         }
         res.build();
         for(int i = 0; i < layers.size(); i++){
             res.locked.set(i, layers.get(i).locked);
             for(int r = 0; r < layers.get(i).weights[i].length; r++){
                 res.biases[i][r] = layers.get(i).biases[r];
                 for(int c = 0; c < res.weights[i][r].length; c++){
                     res.weights[i][r][c] = layers.get(i).weights[r][c];
                 }
             }
         }
         return res;
     }

     public static List<Layer> layerize(nn network, String[] params){
         ArrayList<Layer> res = new ArrayList<>();
         params = new String[2];
         params[0] = network.cost;
         params[1] = network.saveFile;
         for(int i = 0; i < network.weights.length; i++){
             res.add(new Layer(network.weights[i],network.biases[i],network.activations.get(i), network.locked.get(i)));
         }
         return res;
     }

     public static nn deleteLayer(nn before,int index){
         String[] params = new String[2];
         List<Layer> layers = layerize(before, params);
         layers.remove(index);
         return Layer.join(layers, params[0], params[1]);

     }

     public static nn insertLayer(nn before, int index, Layer layer){
         String[] params = new String[2];
         List<Layer> layers = layerize(before, params);
         layers.add(index, layer);
         return Layer.join(layers, params[0], params[1]);
     }
     public static List<nn> split(nn before, int index, String costFirst, String saveFileFirst, String costSecond, String saveFileSecond){
         String params[] = new String[2];
         List<Layer> layers = layerize(before, params);
         List<nn> networks = new ArrayList<>();
         List<Layer> first = new ArrayList<>();
         List<Layer> second = new ArrayList<>();
         for(int i = 0; i < index; i++){
             first.add(layers.remove(0));
         }
         while(layers.size() > 0){
             second.add(layers.remove(0));
         }
         networks.add(Layer.join(first, costFirst, saveFileFirst));
         networks.add(Layer.join(second, costSecond, saveFileSecond));
         return networks;
     }
 }