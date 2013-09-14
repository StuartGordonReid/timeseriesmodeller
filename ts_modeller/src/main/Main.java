/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package main;

/**
 *
 * @author stuart
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static String[] countries = new String[]{
        "/home/stuart/stuartgordonreid@gmail.com/Time series/timeseriesmodeller/ts_modeller/src/data/BRAZIL_train.egb",
        "/home/stuart/stuartgordonreid@gmail.com/Time series/timeseriesmodeller/ts_modeller/src/data/RUSSIA_train.egb",
        "/home/stuart/stuartgordonreid@gmail.com/Time series/timeseriesmodeller/ts_modeller/src/data/INDIA_train.egb",
        "/home/stuart/stuartgordonreid@gmail.com/Time series/timeseriesmodeller/ts_modeller/src/data/CHINA_train.egb",
        "/home/stuart/stuartgordonreid@gmail.com/Time series/timeseriesmodeller/ts_modeller/src/data/SOUTHAFRICA_train.egb"};

    public static void main(String[] args) {
        Simulator sim = new Simulator(countries, 1500);
        sim.start();
    }

}
