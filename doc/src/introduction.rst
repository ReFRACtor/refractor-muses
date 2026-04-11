*****************************
Introduction
*****************************

..
   Getting an underline to work in sphinx is surprisingly
   involved. We have added the role "underline" in the conf.py using
   rst_prolog (so this is included in each file without needing to be
   repeated). We then made a layout.html file in _template that adds a
   local.css file. Finally, the local.css file found in _static
   directory add the underline text decoration.

   One final "gotcha", the block of text for inline markup *must* be separated
   by a non word character. So :underline:`Re`usable will give an *error*
   because there is no separation. You can use a backlash-escaped space which
   is like a no-op in sphinx - it doesn't create a space in the output.

ReFRACtor is a :underline:`Re`\ usable :underline:`FR`\ amework for
:underline:`A`\ tmospheric :underline:`C`\ omposition. refractor-muses
is the integration of ReFRACtor with the TROPESS MUSES system.


      
	    


	   
