```shell
    python quant_test.py --quant_mode calib
```
* To evaluate quantized model, run the following command:
```shell
    python quant_test.py --quant_mode test 
```
When this command finishes, the displayed accuracy is the right accuracy for quantized model. <br> 

* To export xmodel, batch size 1 is must for compilation, and subset_len=1 is to avoid redundant iteration. Run the following command:
```shell
    python quant_test.py --quant_mode test --subset_len 1 --batch_size 1 --deploy
```# Quantize_model
