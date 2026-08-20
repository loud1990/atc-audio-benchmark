# Regional U.S. ATC audio examples

This directory contains 25 paired examples from geographically distinct U.S. areas:

- `reference/`: the normalized real ATC baseline for each scenario
- `degraded/`: one deterministic synthetic degradation of the matching reference

Every file is mono, 16 kHz, signed PCM16 WAV. Durations range from about 68 to 75 seconds.
The reference recordings already contain real radio-channel artifacts; “reference” means the
baseline before this project adds a controlled degradation, not studio-clean speech.

## Pairing and coverage

| ID | Area | Facility | Reference | Degraded example |
|---:|---|---|---|---|
| 001 | New York City, NY | New York TRACON | `ref_001.wav` | `001_bandlimit_mild.wav` |
| 002 | Pompano Beach, FL | PMP Tower | `ref_002.wav` | `002_bandlimit_severe.wav` |
| 003 | Huntsville/Elkmont, AL | HSV West Radar | `ref_003.wav` | `003_static_mild.wav` |
| 004 | Burbank, CA | BUR Local Control | `ref_004.wav` | `004_static_moderate.wav` |
| 005 | Houston/Brookshire, TX | TME Tower | `ref_005.wav` | `005_static_severe.wav` |
| 006 | San Diego, CA | San Diego-area ATC | `ref_006.wav` | `006_white_noise.wav` |
| 007 | Newark/Colonia, NJ | New York TRACON N90 | `ref_007.wav` | `007_pink_noise.wav` |
| 008 | Asheville/Fletcher, NC | KAVL Ground/Tower/Approach | `ref_008.wav` | `008_vhf_bandlimited_noise.wav` |
| 009 | St. Augustine, FL | Local Control | `ref_009.wav` | `009_fading_slow.wav` |
| 010 | Kissimmee/Micanopy, FL | ISM | `ref_010.wav` | `010_fading_severe.wav` |
| 011 | Oklahoma City, OK | Tower Control | `ref_011.wav` | `011_dropouts_short.wav` |
| 012 | Kellyton, AL | Alabama-area ATC | `ref_012.wav` | `012_dropouts_heavy.wav` |
| 013 | Dayton, VA | Virginia-area ATC | `ref_013.wav` | `013_beginning_clipped.wav` |
| 014 | Alpharetta, GA | Atlanta-area ATC replay | `ref_014.wav` | `014_ending_clipped.wav` |
| 015 | Duluth, MN | Duluth ATC | `ref_015.wav` | `015_hard_clipping.wav` |
| 016 | Portland/Arundel, ME | PWM Radar | `ref_016.wav` | `016_soft_saturation.wav` |
| 017 | Chandler, AZ | Local Control | `ref_017.wav` | `017_agc_pumping.wav` |
| 018 | Worcester/Leicester, MA | ORH ATC | `ref_018.wav` | `018_static_pops_crackles.wav` |
| 019 | Knoxville, TN | TYS Local Control | `ref_019.wav` | `019_electrical_hum.wav` |
| 020 | Wilmington, DE | ILG Local Control | `ref_020.wav` | `020_interference_tone.wav` |
| 021 | Sarasota, FL | SRQ Local Control | `ref_021.wav` | `021_adjacent_channel_atc.wav` |
| 022 | Seligman, AZ | Area F Sector 8 | `ref_022.wav` | `022_opus_low_bitrate.wav` |
| 023 | Louisville, KY | SDF Local Control East | `ref_023.wav` | `023_mp3_repeated_low_bitrate.wav` |
| 024 | Fairbanks, AK | FAI Local Control | `ref_024.wav` | `024_packet_loss.wav` |
| 025 | Maui/Kaupo, HI | Hawaii Local Control | `ref_025.wav` | `025_weak_vhf_compound.wav` |

The collection spans 25 areas in 19 states. Geography was selected to broaden the pool of
regional speech patterns, but a recording’s location does not establish any individual
speaker’s accent or dialect. No speaker-level dialect labels are assigned.

## Provenance and reuse

The source URLs, facilities, preparation settings, and rights notes are recorded in
[`../../configs/regional_sources_v1.yaml`](../../configs/regional_sources_v1.yaml). The first
source is an FAA recording mirrored by Wikimedia Commons as U.S. public-domain material; the
other sources are FAA ATC recordings published in NTSB public dockets. The WAV files here are
trimmed/normalized derivatives, and the files in `degraded/` contain additional synthetic
processing.

The exact scenario definitions and deterministic seed are in
[`../../configs/personal_regional_v1.yaml`](../../configs/personal_regional_v1.yaml).
