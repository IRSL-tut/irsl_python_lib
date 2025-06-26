from scipy.signal import iirfilter
import numpy as np

class IIRFilter(object):
    def __init__(self, N, Wn, fs, btype='lowpass', **kwargs):
        """
        Initializes the IIR filter with the given parameters.

        Args:
            N (int): The order of the filter.
            Wn (float): Critical frequency or frequencies. [Hz]
            fs (float): The sampling frequency of the digital system. [Hz]
            btype (str, optional): The type of filter to design. Options are 
                'lowpass', 'highpass', 'bandpass', or 'bandstop'. Defaults to 'lowpass'.
            **kwargs: Additional keyword arguments to pass to the `iirfilter` function.

        Attributes:
            b (ndarray): Numerator polynomial coefficients of the IIR filter.
            a (ndarray): Denominator polynomial coefficients of the IIR filter.
            N (int): The order of the filter.
        """
        self.b, self.a = iirfilter(N, Wn, fs=fs, btype=btype, **kwargs)
        self.N = N
        self.reset()
        #self.prev_inputs = [0] * ( self.N + 1)
        #self.prev_outputs = [0] * ( self.N + 1)
    #def filter(self, val):
    #    """
    #    Apply the IIR filter to the input value.
    #    """
    #    # Shift previous inputs and outputs
    #    self.prev_inputs = [val] + self.prev_inputs[:-1]
    #    output = self.b[0] * self.prev_inputs[0]
    #    for i in range(1, self.N+1):
    #        output += self.b[i] * self.prev_inputs[i] - self.a[i] * self.prev_outputs[i - 1]
    #    output /= self.a[0]
    #    # Shift previous outputs
    #    self.prev_outputs = [output] + self.prev_outputs[:-1]
    #    return output
    def filter(self, val):
        """
        Applies an IIR (Infinite Impulse Response) filter to the input value

        Args:
            val (float): The current input value to be filtered.

        Returns:
            float: The filtered output value.

        Note:
            Using the Direct Form-II implementation.
            The filter uses coefficients `a` and `b` to compute the filtered output 
            based on the current input value and previous values stored in `prev_values`.

        References:
            - https://en.wikipedia.org/wiki/Digital_filter
        """
        ## IIRFilter implementation based on DirectForm-II.
        ## Cf. https://en.wikipedia.org/wiki/Digital_filter
        feedback = self.a[0] * val
        for i in range(self.N):
            feedback -= self.a[i + 1] * self.prev_values[i]
        filtered = self.b[0] * feedback
        for i in range(self.N):
            filtered += self.b[i + 1] * self.prev_values[i]
        # Shift previous values
        self.prev_values = [feedback] +  self.prev_values[:-1]
        return filtered
    def reset(self, val=0.0):
        """
        Resets the internal state of the filter to a specified value.

        Args:
            val (float, default=0.0): The value to reset the filter's internal state to.

        """
        self.prev_values = [val] * (self.N + 1)

class VectorFilter(object):
    def __init__(self, size, N, Wn, fs, **kwargs):
        """
        Initializes a collection of IIRFilter instances.

        Args:
            size (int): The size of vector to be filtered
            N (int): The order of the filter.
            Wn (float or array_like): The critical frequency or frequencies. 
                For digital filters, this is normalized from 0 to 1, where 1 is the Nyquist frequency.
            fs (float): The sampling frequency of the digital system.
            **kwargs: Additional keyword arguments to be passed to the IIRFilter constructor.

        Attributes:
            filters (list): A list containing the created IIRFilter instances.
            size (int): The number of filters in the collection.
        """
        self.filters = []
        self.size = size
        for i in range(size):
            self.filters.append( IIRFilter(N, Wn, fs, **kwargs) )

    def filter(self, vec):
        """
        Applies the filter to the input vector.

        Args:
            vec (numpy.ndarray): Input vector of size `self.size` containing the
                values to be filtered.

        Returns:
            numpy.ndarray: A vector of the same size as `vec`, where each element
            is the result of applying the corresponding filter to the input vector.
        """
        res = np.zeros(self.size)
        for i in range(self.size):
            res[i] = self.filters[i].filter(vec[i])
        return res

    def filter_inplace(self, vec):
        """
        Applies the filter to the input vector. Change the values in the vector

        Args:
            vec (numpy.ndarray): Input vector of size `self.size` containing the
                values to be filtered.

        """
        for i in range(self.size):
            vec[i] = self.filters[i].filter(vec[i])

    def filterList(self, lst):
        """
        Applies a list of filters to a corresponding list of input values.

        This method takes a list of input values and applies each filter in 
        `self.filters` to the corresponding value in the input list. The 
        filtering is performed using the `filter` method of each filter object.

        Args:
            lst (list): A list of input values to be filtered. The length of 
                        this list should match the length of `self.filters`.

        Returns:
            list: A list of filtered values, where each value is the result 
                  of applying the corresponding filter to the input value.

        Raises:
            ValueError: If the length of `lst` does not match the length of 
                        `self.filters`.
        """
        return [ f.filter(l) for l, f in  zip(lst, self.filters) ]

    def reset(self, vec=None):
        """
        Resets the state of the filters.

        Args:
            vec (list, optional): A list of values to reset each filter with. 
                If None, filters are reset without specific values.
        """
        if vec is not None:
            for v, f in zip(vec, self.filters):
                f.reset(v)
        else:
            for f in self.filters:
                f.reset()

#import matplotlib.pyplot as plt
#import irsl.filter.iirfilter
#import numpy as np
#from irsl.filter.iirfilter import IIRFilter
#from irsl.filter.iirfilter import VectorFilter
#if True:
#    # Example usage of IIRFilter
#    fs = 1000.0  # Sampling frequency in Hz
#    N = 4       # Filter order
#    Wn = 100.0   # Cutoff frequency in Hz
#    duration = 0.5  # Signal duration in seconds
#    t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time vector
#    signal = np.sin(2 * np.pi * 25.0 * t) + 0.4 * np.sin(2 * np.pi * 200.0 * t)  # Example signal
#    iir_filter = IIRFilter(N, Wn, fs, btype='lowpass')
#    filtered_signal = np.array([iir_filter.filter(val) for val in signal])
#    # Plot the original and filtered signals
#    plt.figure()
#    plt.plot(t, signal, label="Original Signal")
#    plt.plot(t, filtered_signal, label="Filtered Signal", linestyle='--')
#    plt.xlabel("Time [s]")
#    plt.ylabel("Amplitude")
#    plt.title("IIR Filter Example")
#    plt.legend()
#    plt.grid()
#    plt.show()
#if True:
#    # Example usage of VectorFilter
#    fs = 1000.0  # Sampling frequency in Hz
#    N = 3       # Filter order
#    Wn = 250.0   # Cutoff frequency in Hz
#    duration = 0.5  # Signal duration in seconds
#    vector_size = 3
#    vector_filter = VectorFilter(vector_size, N, Wn, fs, btype='lowpass')
#    vector_signal = np.array([ np.sin(2 * np.pi * 60.0 * t), np.sin(2 * np.pi * 120.0 * t), np.sin(2 * np.pi * 200.0 * t) ]).T
#    filtered_vector_signal = np.array([ vector_filter.filter(vec) for vec in vector_signal ])
#    # Plot the original and filtered vector signals
#    plt.figure()
#    for i in range(vector_size):
#        plt.subplot(vector_size, 1, i + 1)
#        plt.plot(t, vector_signal[:, i], label=f"Original Signal {i+1}")
#        plt.plot(t, filtered_vector_signal[:, i], label=f"Filtered Signal {i+1}", linestyle='--')
#        plt.xlabel("Time [s]")
#        plt.ylabel("Amplitude")
#        plt.title(f"Signal {i+1}")
#        plt.legend()
#        plt.grid()
#    plt.tight_layout()
#    plt.show()
