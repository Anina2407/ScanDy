# Different versions
All fitting on saccade amplitude and foveation duration. 
All start: random. 

* fade_2 means gradually fades feature maps from original to 1 starting at frame i, with i being 5 or 15
--- 
steps = np.ones(viddata.feature_maps.shape[0])
steps[i:] = np.linspace(1, 0, viddata.feature_maps.shape[0] - i)  
viddata.feature_maps = viddata.feature_maps * steps[:, None, None] + 1 * (1 - steps[:, None, None])
---

* fade_3 time-varying saliency weighting to feature maps - exponentially rises from baseline to a peak  (here at frame 18, but variable) then exponentially decays back to baseline

