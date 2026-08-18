# Brand images

Home Assistant does not read an icon out of a custom integration. The frontend
builds every brand image URL from `brands.home-assistant.io`, so until a domain
has an entry in [home-assistant/brands][brands] it is drawn with the generic
placeholder — no file placed inside `custom_components/needle_assist/` changes
that.

These are the files that entry needs, already at the required sizes:

| file | size | ink |
|---|---|---|
| `icon.png` | 256×256 | black |
| `icon@2x.png` | 512×512 | black |
| `logo.png` | 256×256 | black |
| `logo@2x.png` | 512×512 | black |
| `dark_icon.png`, `dark_icon@2x.png` | 256×256, 512×512 | white |
| `dark_logo.png`, `dark_logo@2x.png` | 256×256, 512×512 | white |

PNG, trimmed to the drawing, transparent background, no metadata. The mark is a
square drawing, so the icon and the logo are the same image — `logo.png` only
has to keep its shortest side between 128 and 256 pixels, which 256×256 does.

The artwork is a black line drawing. Its alpha channel is the inverse of the
luminance rather than a colour key, so the anti-aliased edges stay soft instead
of turning into a grey halo against a dark theme. The `dark_` variants are the
same alpha with white ink, because pure black is invisible on Home Assistant's
dark theme.

## Submitting

The brands repository is separate and takes its own pull request:

1. Fork [home-assistant/brands][brands].
2. Copy `custom_integrations/needle_assist/` from here into the fork, at the
   same path.
3. Open a pull request. Custom integrations go under `custom_integrations/`,
   never `core_integrations/`.

Once it is merged the icon appears in Settings → Devices & services, in the
integration picker and in HACS. Nothing in this repository has to change, and
no release is needed.

While there is no entry, the `hacs` job in `.github/workflows/validate.yml`
passes only because it is told `ignore: brands`. That line can be deleted after
the pull request is merged.

[brands]: https://github.com/home-assistant/brands
