"""
High-speed, half-resolution demosaicing algorithm for Raspberry Pi camera.

This is designed to be a work-alike for picamera.array.PiBayerArray, but
much faster.
"""
import numpy as np
import picamera.array

class PiFastBayerArray(picamera.array.PiBayerArray):
    """
    Produces a 3-dimensional RGB array from raw Bayer data, at half resolution.

    This output class should be used with the :meth:`~picamera.PiCamera.capture`
    method, with the *bayer* parameter set to ``True`` (this includes the raw
    Bayer data in the JPEG metadata).  This class extracts the Bayer data, and
    stores it in the :attr:`~PiArrayOutput.array` attribute.  The raw Bayer array
    is made of packed 10-bit values, where every fifth byte contains the least
    significant bits.  It can be accessed as shown::
    
            import picamera
            import picamera.array
            
            with picamera.PiCamera() as camera:
                with picamera.array.PiFastBayerArray(camera) as output:
                    camera.capture(output, 'jpeg', bayer=True)
                    print(output.array.shape)
    
    As with :class:`~PiBayerArray`, note that this output is always at full
    resolution, regardless of the camera's resolution setting.  
    
    In many situations, it is desirable to convert the raw array into an 8-bit
    RGB array.  This can be done with the :meth:`demosaic` method.  As in the
    superclass, this method converts the raw array to an RGB array.  However,
    this class uses a much faster (and much cruder) demosaicing algorithm, so
    each pixel in the array is based on a group of four pixels (red, blue, and 
    two green) on the sensor.  This produces a half-resolution RGB array, with
    much less processing time required.

    See :class:`~PiBayerArray` for full-resolution conversion to RGB.  There
    are a few important differences between this class and that:
    * The resolution will always be half that used by ``PiBayerArray``
    * The :attr:`~PiFastBayerArray.array` attribute contains raw Bayer data, not
      unpacked RGB data.
    * The output of :meth:`~demosaic` will have half the resolution compared to
      :meth:`~PiBayerArray` but will still be an RGB array of unsigned 8-bit 
      integers.
    """
    def flush(self):
        picamera.array.PiArrayOutput.flush(self) 
        self._demo = None
        ver = 1
        data = self.getvalue()[-6404096:]
        if data[:4] != b'BRCM':
            ver = 2
            data = self.getvalue()[-10270208:]
            if data[:4] != b'BRCM':
                raise PiCameraValueError('Unable to locate Bayer data at end of buffer')
        # Strip header
        data = data[32768:]
        # Reshape into 2D pixel values
        reshape, crop = {
            1: ((1952, 3264), (1944, 3240)),
            2: ((2480, 4128), (2464, 4100)),
            }[ver]
        data = np.frombuffer(data, dtype=np.uint8).\
                reshape(reshape)[:crop[0], :crop[1]]
        self.array = data # This is not quite the raw Bayer data - every 5th element
          # is four lots of two least-significant-bits.

    def demosaic(self, shift=0):
        """Convert the raw Bayer data into a half-resolution RGB array.

        This uses a really blunt demosaicing algorithm: group pixels in squares,
        and then use the red, blue, and two green pixels from each square to
        calculate an RGB value.  This is calculated as three unsigned 8-bit 
        integers.

        As the sensor is 10 bit but output is 8-bit, we provide the ``shift`` 
        parameter.  Setting this to 2 will return the lower 8 bits, while setting 
        it to 0 (the default) will return the upper 8 bits.  In the future, 
        there may be an option to work in 16-bit integers and return all of them 
        (though that would be slower).  Currently, if ``shift`` is nonzero and
        some pixels have higher values than will fit in the 8-bit output, overflow
        will occur and those pixels may no longer be bright - so use the ``shift`` 
        argument with caution.
        
        NB that the highest useful ``shift`` value is 3; while the sensor is only 
        10-bit, there are two green pixels on the sensor for each output pixel.
        Thus, we gain an extra bit of precision from averaging, allowing us to
        effectively produce an 11-bit image.
        """
        if self._demo is None:
            # XXX Again, should take into account camera's vflip and hflip here
            # Extract the R, G1, G2, B pixels into separate slices
            # NB these should _not_ need to be copied at this stage.
            # NB we end up with odd and even arrays for each because every
            # 5th element is the least-significant-bits.
            def bayer_slices(i, j, shift=shift):
                if shift == 0: # Return the top 8 bits
                    # This should be really fast - they ought to be slices
                    return self.array[i::2, j::5], self.array[i::2,j+2::5]
                else:
                    # Left-shift the arrays so we can fill the LSB later
                    a, b = bayer_slices(i, j, shift=0)
                    a, b = a << shift, b << shift # NB this copies a, b
                    # Now retrieve and add in the two least significant bits
                    # These are stored, packed, in every 5th byte:
                    lsb = self.array[i::2, 4::5]
                    # The LSB will be in bits (3-j)*2 and (3-j)*2 + 1
                    if shift == 2:
                        a += (lsb >> (3 - j)*2) & 3
                        b += (lsb >> (1 - j)*2) & 3
                    elif shift == 1:
                        a += (lsb >> ((3 - j)*2 + 1)) & 1
                        b += (lsb >> ((1 - j)*2 + 1)) & 1
                    elif shift == 3:
                        # A shift of 3 leaves the LSB at zero.  It's only
                        # included because the two green pixels means that
                        # we do generate an LSB for green even with a shfit
                        # of 3.  This might be handy for fluorescence.
                        a += ((lsb >> (3 - j)*2) & 3) << 1
                        b += ((lsb >> (1 - j)*2) & 3) << 1
                    return a, b
            Ra, Rb = bayer_slices(1,0)
            G1a, G1b = bayer_slices(0,0)
            G2a, G2b = bayer_slices(1,1)
            Ba, Bb = bayer_slices(0,1)
            # Make an array of the right size
            shape = (Ra.shape[0], Ra.shape[1] * 2, 3)
            rgb = np.empty(shape, dtype=Ra.dtype)
            # Now put the relevant values in
            rgb[:,0::2,0] = Ra # Red pixels (even)
            rgb[:,1::2,0] = Rb # Red pixels (odd)
            rgb[:,0::2,2] = Ba
            rgb[:,1::2,2] = Bb
            rgb[:,0::2,1] = G1a/2 # There are twice as many greens, so we
            rgb[:,1::2,1] = G1b/2 # take an average
            rgb[:,0::2,1] += G2a/2
            rgb[:,1::2,1] += G2b/2
            self._demo = rgb
        return self._demo

