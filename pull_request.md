## ✨ Description

We release the JETS (Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech) model in Amphion. JETS has a simplified training pipeline and outperforms a cascade of separately learned models. Specifically, JETS is jointly trained FastSpeech2 and HiFi-GAN with an alignment module.
How to test: see egs/Jets/README.md
Major contribution for this PR: @hansheng-zhang @chenjianzhen666 @So1a

## 👨‍💻 Changes Proposed

- [ ] Add the Jets model in the tts section
- [ ] Add jets' versioin of mpd and msd in vocoders/gan/discriminator

## 🧑‍🤝‍🧑 Who Can Review?

## ✅ Checklist

- [ ] Code has been reviewed
- [ ] Code complies with the project's code standards and best practices
- [ ] Code has passed all tests
- [ ] Code does not affect the normal use of existing features
- [ ] Code has been commented properly
- [ ] Documentation has been updated (if applicable)
- [ ] Demo/checkpoint has been attached (if applicable)
