# Fibonacci Stretch

Stretch the rhythm of an audio track along the Fibonacci sequence.

This README pertains to the `fibonaccistretch` Python module; for information about the Groovinator VST/AU plugin, please see [this repository](https://github.com/usdivad/Groovinator).

## Examples
http://usdivad.com/fibonaccistretch/examples.html

## Usage
Fibonacci stretch the first 90 seconds of Michael Jackson's "Human Nature" by a factor of 1 using input rhythm `[1,0,0,1,0,0,1,0]`:

```python
import fibonaccistretch as fib

fib.fibonacci_stretch_track("data/humannature_90s.mp3",
                            original_rhythm=[1,0,0,1,0,0,1,0],
                            stretch_factor=1,
                            overlay_clicks=True)
```

Take a listen to the [result](data/out_humannature_90s_stretched.mp3).

## More info
See [fibonaccistretch.ipynb](nbs/fibonaccistretch.ipynb) for a more detailed writeup and implementation (note: the outputs are cleared, so to see figures and listen to audio you'll need to run the notebook using Jupyter). A static HTML version is also available [here](http://usdivad.com/fibonaccistretch).

Presented at the [2017 AES Workshop on Intelligent Music Production](http://www.semanticaudio.co.uk/events/wimp2017/).