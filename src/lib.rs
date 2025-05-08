use nih_plug::prelude::*;
use nih_plug::util::StftHelper;
use realfft::num_complex::ComplexFloat;
use realfft::num_traits::Zero;
use realfft::{num_complex::Complex, ComplexToReal, RealFftPlanner, RealToComplex};
use std::convert::From;
use std::sync::Arc;

const WINDOW_SIZE: usize = 4096;
const FILTER_SIZE: usize = WINDOW_SIZE / 2;
const FFT_WINDOW_SIZE: usize = WINDOW_SIZE + FILTER_SIZE;

struct InverseHarmony {
    params: Arc<InverseHarmonyParams>,
    stft: StftHelper,
    r2c_plan: Arc<dyn RealToComplex<f32>>,
    c2r_plan: Arc<dyn ComplexToReal<f32>>,
    freq_buffer: Vec<f32>,
    sample_rate: f32,
    r2c_input_buffer: Vec<f32>,
    r2c_output_buffer: Vec<Complex<f32>>,
    c2r_input_buffer: Vec<Complex<f32>>,
    freq_step: f32,
    scratch_complex_buffer: Vec<Complex<f32>>,
    c2r_len: usize,
    c2r_output_buffer: Vec<f32>,
    filter: Vec<f32>,
}

// TODO: A lot of the FFT parameters that would be really nice to have
// adjustable by the user are quite costly to have in each buffer and really
// should be reinstantiated upon change and not per FFT-window while live
// processing.
#[derive(Params)]
struct InverseHarmonyParams {
    /// The parameter's ID is used to identify the parameter in the wrappred plugin API. As long as
    /// these IDs remain constant, you can rename and reorder these fields as you wish. The
    /// parameters are exposed to the host in the same order they were defined. In this case, this
    /// gain parameter is stored as linear gain while the values are displayed in decibels.
    // #[id = "gain"]
    // pub gain: FloatParam,

    #[id = "f0"]
    pub f0: FloatParam,

    #[id = "auto-f0"]
    pub autof0: BoolParam,

    #[id = "f0-with-fft"]
    pub f0withfft: BoolParam,

    // Measured in ms
    #[id = "f0window"]
    pub f0window: IntParam,

    #[id = "dry/wet"]
    pub dry_wet: FloatParam,
}

impl Default for InverseHarmony {
    fn default() -> Self {
        let mut planner = RealFftPlanner::new();
        // Forward FFT
        let r2c_plan = planner.plan_fft_forward(FFT_WINDOW_SIZE);
        // Inverse FFT
        let c2r_plan = planner.plan_fft_inverse(FFT_WINDOW_SIZE);
        // Allocate buffers for convolution filter
        let r2c_input_buffer = r2c_plan.make_input_vec();
        let r2c_output_buffer = r2c_plan.make_output_vec();
        let c2r_input_buffer = r2c_output_buffer.clone();
        let c2r_len = c2r_input_buffer.len();
        let freq_buffer = vec![0.0f32; r2c_output_buffer.len()];
        let scratch_complex_buffer = c2r_plan.make_scratch_vec();
        let c2r_output_buffer = c2r_plan.make_output_vec();

        // Build a super simple low-pass filter from one of the built in window functions
        let filter = util::window::hann(FFT_WINDOW_SIZE);
        // // And make sure to normalize this so convolution sums to 1
        // let filter_normalization_factor = filter.iter().sum::<f32>().recip();
        // for sample in &mut filter {
        //     *sample *= filter_normalization_factor;
        // }

        // r2c_input_buffer[..FILTER_SIZE].copy_from_slice(&filter_window);

        // r2c_plan
        //     .process_with_scratch(&mut r2c_input_buffer, &mut r2c_output_buffer, &mut [])
        //     .unwrap();

        Self {
            params: Arc::new(InverseHarmonyParams::default()),
            stft: util::StftHelper::new(2, WINDOW_SIZE, FILTER_SIZE),
            r2c_plan,
            c2r_plan,
            sample_rate: 44100.0,
            freq_buffer,
            r2c_input_buffer,
            r2c_output_buffer: r2c_output_buffer.clone(),
            c2r_input_buffer,
            freq_step: 0.0,
            scratch_complex_buffer,
            c2r_len,
            c2r_output_buffer,
            filter,
        }
    }
}

impl Default for InverseHarmonyParams {
    fn default() -> Self {
        Self {
            f0: FloatParam::new(
                "f0",
                440.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            ),

            autof0: BoolParam::new("Detect f0", true),

            f0withfft: BoolParam::new("Calculate f0 on every FFT window", true),

            // f0 will be calculated at the start of the f0window and then
            // stored as the axis on which to reflect
            // Value in ms
            // Only
            f0window: IntParam::new(
                "f0 window",
                1000,
                IntRange::Linear {
                    min: 100,
                    max: 1000000,
                },
            ),

            dry_wet: FloatParam::new("Dry/Wet", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 }),
        }
    }
}

impl Plugin for InverseHarmony {
    const NAME: &'static str = "Inverse Harmony";
    const VENDOR: &'static str = "Ethan Meltzer";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "emeltzer@oberlin.edu";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // The first audio IO layout is used as the default. The other layouts may be selected either
    // explicitly or automatically by the host or the user depending on the plugin API/backend.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    // If the plugin can send or receive SysEx messages, it can define a type to wrap around those
    // messages here. The type implements the `SysExMessage` trait, which allows conversion to and
    // from plain byte buffers.
    type SysExMessage = ();
    // More advanced plugins can use this to run expensive background tasks. See the field's
    // documentation for more information. `()` means that the plugin does not have any background
    // tasks.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        context.set_latency_samples(self.stft.latency_samples() + (FILTER_SIZE as u32 / 2));
        self.sample_rate = buffer_config.sample_rate;
        // Frequencies go up to nyquist
        self.freq_step = (self.sample_rate / 2.0) / (self.freq_buffer.len() as f32);
        for i in 0..self.freq_buffer.len() {
            self.freq_buffer[i] = i as f32 * self.freq_step;
        }
        self.stft.set_block_size(WINDOW_SIZE);

        true
    }

    fn reset(&mut self) {
        self.stft.set_block_size(WINDOW_SIZE);
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.stft
            .process_overlap_add(buffer, 1, |_channel_idx, real_fft_buffer| {
                // Need to zero out the c2r input buffer every loop
                self.c2r_input_buffer[..].fill(Complex::zero());

                // Copy values into dedicated dry buffer, input buffer to r2c will be destroyed
                self.r2c_input_buffer[..].copy_from_slice(&real_fft_buffer);
                // for (i, val) in real_fft_buffer.iter().enumerate() {
                //     self.r2c_input_buffer[i] = *val;
                // }

                // nih_log!("Input");
                // nih_dbg!(&real_fft_buffer);

                // Get max value of input buffer
                let mut gain_scale = 0.0;
                for el in &mut *real_fft_buffer {
                    let el_abs = el.abs();
                    if el_abs > gain_scale {
                        gain_scale = el_abs;
                    }
                }

                // Forward FFT, `real_fft_buffer` already is padded with zeros, and
                // the padding from the last iteration will already have been added
                // back to the start of the buffer
                self.r2c_plan
                    .process_with_scratch(
                        real_fft_buffer,
                        &mut self.r2c_output_buffer,
                        &mut self.scratch_complex_buffer,
                    )
                    .unwrap();

                // Getting our most present frequency
                let mut max: (f32, usize) = (0.0, 0);
                for (index, elem) in self.r2c_output_buffer.iter().enumerate() {
                    let new = elem.re();
                    if new > max.0 {
                        max = (new, index);
                    }
                }

                // We didn't find one, silence
                // We want to early return in this case
                if max.1 == 0 {
                    return;
                }

                // nih_log!("Dry after FFT forward: {:#?}", &real_fft_buffer);

                // nih_log!("Complex pre-processing");
                // nih_dbg!(&self.r2c_output_buffer);

                // Find its actual frequency value
                let f0 = match self.params.autof0.value() {
                    true => self.freq_buffer[max.1],
                    false => self.params.f0.value(),
                };
                let f02 = f0 * f0;
                // nih_log!("f0: {f0}");

                // Invert frequency
                for (i, a) in self.freq_buffer.iter().enumerate() {
                    // Not going to deal with bins greater than highest key on a piano, leads to too much energy concentrated in lowest
                    // bins
                    // if *a > 4186.01 {
                    //     break;
                    // }

                    let mut inv = *a;

                    // Throw out the top and bottom bins to avoid weird artifacting
                    // if inv == 0.0 || inv == self.freq_buffer[self.freq_buffer.len() - 1] {
                    //     continue;
                    // }
                    inv = f02 / inv;

                    // Binary search freq_buffer for inverse
                    // Need the bins at the returned index and one less
                    let bin_index = match self
                        .freq_buffer
                        .binary_search_by(|probe| probe.total_cmp(&inv))
                    {
                        Ok(e) => e,
                        Err(e) => e,
                    };

                    // If the bin index doesn't exist, then just set to the highest index bin
                    let upper_bin_freq = match self.freq_buffer.get(bin_index) {
                        Some(f) => *f,
                        None => self.freq_buffer[self.freq_buffer.len() - 1],
                    };

                    let delta = upper_bin_freq - inv;
                    // Edge cases where we're dealing with uppermost or lowermost bin
                    if delta < 0.0 {
                        self.c2r_input_buffer[bin_index - 1] += self.r2c_output_buffer[i];
                        continue;
                    }

                    if bin_index == 0 {
                        self.c2r_input_buffer[bin_index] += self.r2c_output_buffer[i];
                        continue;
                    }

                    // Linear bin interpolation, get "closeness" of freq value
                    // to adjacent bins

                    // Between 0 and 1
                    let norm_del = delta / self.freq_step;
                    self.c2r_input_buffer[bin_index] += norm_del * self.r2c_output_buffer[i];
                    self.c2r_input_buffer[bin_index - 1] +=
                        (1.0 - norm_del) * self.r2c_output_buffer[i];
                }

                // Windowing
                // for (bin, filterval) in self
                //     .c2r_input_buffer
                //     .iter_mut()
                //     .zip(&self.lp_filter_spectrum)
                // {
                //     *bin *= filterval;
                // }

                // All of the very small components of energy in higher bins
                // get added to bin 0 resulting in huge gain. Easiest thing
                // to do is discard bin 0 entirely.
                // Need to set imaginary parts of first and last index to 0.
                self.c2r_input_buffer[0].im = 0.0;
                self.c2r_input_buffer[self.c2r_len - 1].im = 0.0;

                // nih_log!("Before back");
                // nih_dbg!(&self.c2r_input_buffer);

                self.c2r_plan
                    .process_with_scratch(
                        &mut self.c2r_input_buffer,
                        &mut self.c2r_output_buffer,
                        &mut self.scratch_complex_buffer,
                    )
                    .unwrap();

                // Normalize gain to unity with input
                let mut out_scale = 0.0;
                for ele in &mut self.c2r_output_buffer {
                    let ele_abs = ele.abs();
                    if ele_abs > out_scale {
                        out_scale = ele_abs;
                    }
                }

                let factor = gain_scale / out_scale;
                for ele in &mut self.c2r_output_buffer {
                    *ele *= factor;
                }

                nih_log!("INVSTFT Post-normal: {:#?}", self.c2r_output_buffer);

                // Apply hann filter here to get rid of time aliasing
                for (out_val, fil_val) in self.c2r_output_buffer.iter_mut().zip(&self.filter) {
                    *out_val *= fil_val;
                }

                // nih_log!("Dry_wet: {}", self.params.dry_wet.value());
                // nih_log!("Before dry/wet WET: {:#?}", self.c2r_output_buffer);
                // nih_log!("Before dry/wet DRY: {:#?}", &real_fft_buffer);
                // Combine dry and wet
                // TODO: getting some values way out of gain range here, why is that?
                for (i, e) in self.r2c_input_buffer.iter().enumerate() {
                    real_fft_buffer[i] = ((1.0 - self.params.dry_wet.value()) * *e)
                        + (self.params.dry_wet.value() * self.c2r_output_buffer[i]);
                }

                // nih_log!("Output");
                // nih_dbg!(&real_fft_buffer);
            });

        ProcessStatus::Normal
    }
}

impl ClapPlugin for InverseHarmony {
    const CLAP_ID: &'static str = "dev.everwild.inverse_harmony";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Uses FFT to estimate f0 and uses that as an axis on which to flip in the frequency domain.");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

impl Vst3Plugin for InverseHarmony {
    const VST3_CLASS_ID: [u8; 16] = *b"Inverse_Harmony.";

    // And also don't forget to change these categories
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nih_export_clap!(InverseHarmony);
nih_export_vst3!(InverseHarmony);
