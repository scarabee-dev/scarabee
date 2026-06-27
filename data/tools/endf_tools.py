import endf
import numpy as np

# Silences warnings about ignores sections from endf library
# Mostly just covariances, so we don't care about these
import warnings
warnings.simplefilter("ignore")


def get_fission_yield_function(mat: endf.Material) -> np.polynomial.Polynomial | endf.Tabulated1D:
    """
    Creates a function-like object to evaluate the total fission yield at any
    incident neutron energy (in units of eV).

    Parameters
    ----------
    mat : endf.Material
        Fissionable material with an MF 1 MT 452 section.

    Returns
    -------
    np.polynomial.Polynomial | endf.Tabulated1D
        A function-like object to evaluate the fission yield.

    Raises
    ------
    KeyError
        If the material does not contain MF 1 MT 451.
    RuntimeError
        If and unknown fission-yield representation is found.
    """
    if not (1, 452) in mat:
        raise KeyError("Material does not have MF 1 MT 452")
    mf1mt452 = mat[1, 452]
    LNU = mf1mt452['LNU']
    if LNU == 1:
        # Polynomial
        return np.polynomial.Polynomial(mf1mt452['C'])
    elif LNU == 2:
        # Tabulated
        return mf1mt452['nu']
    else:
        raise RuntimeError(f"Unknown value of LNU = {LNU}")

def get_fission_energy_release_components(mat: endf.Material) -> dict:
    """
    Creates a dictionary of function-like object to evaluate the different
    components of the energy released from fission at any incident neutron
    energy (in units of eV).

    The returned dictionary should have the following keys:
    ['EFR', 'ENP', 'END', 'EGP', 'EGD', 'EB', 'ENU', 'ER', 'ET'].

    Parameters
    ----------
    mat : endf.Material
        Fissionable material with an MF 1 MT 458 section.

    Returns
    -------
    dict
        Keys are strings for the components, with each entry being a
        function-like object to evaluate the fission yield.

    Raises
    ------
    KeyError
        If the material does not contain MF 2 MT 151.
    TypeError
        If and unknown fission-yield representation is found.
    RuntimeError
        If no polynomial coefficients are provided for an energy component.
    """
    if not (1, 458) in mat:
        raise KeyError("Material does not have MF 1 MT 458")
    mf1mt458 = mat[1, 458]
    nu = get_fission_yield_function(mat)
    nu_0 = nu(1.E-5)
    
    components = ('EFR', 'ENP', 'END', 'EGP', 'EGD', 'EB', 'ENU', 'ER', 'ET')
    
    # Fill initial output dictionary with the basic polynomial representation
    out = {}
    for comp in components:
        # Get the polynomial form
        if isinstance(mf1mt458[comp], list):
            coeffs = []
            for c in mf1mt458[comp]:
                coeffs.append(c[0])

            if len(coeffs) > 1:
                out[comp] = np.polynomial.Polynomial(coeffs)
            elif len(coeffs) == 1:
                # Should use Sher-Beck formulas if we only have a constant
                C = coeffs[0]
                match comp:
                    case 'ET':
                        dlt = lambda E, nu=nu, nu_0=nu_0: -1.057*E + 8.07E6*(nu(E) - nu_0)
                        out[comp] = lambda E, C=C, dlt=dlt: C - dlt(E)
                    case 'EB':
                        dlt = lambda E : 0.075 * E
                        out[comp] = lambda E, C=C, dlt=dlt: C - dlt(E)
                    case 'EGD':
                        dlt = lambda E : 0.075 * E
                        out[comp] = lambda E, C=C, dlt=dlt: C - dlt(E)
                    case 'ENU':
                        dlt = lambda E : 0.100 * E
                        out[comp] = lambda E, C=C, dlt=dlt: C - dlt(E)
                    case 'EFR':
                        out[comp] = np.polynomial.Polynomial(coeffs)
                    case 'ENP':
                        dlt = lambda E, nu=nu, nu_0=nu_0: -1.307*E + 8.07E6*(nu(E) - nu_0)
                        out[comp] = lambda E, C=C, dlt=dlt: C - dlt(E)
                    case 'EGP':
                        out[comp] = np.polynomial.Polynomial(coeffs)
                    case 'END':
                        out[comp] = np.polynomial.Polynomial(coeffs)
                    case 'ER':
                        # We skip ER (recoverable energy) because we will reconstruct
                        # it after as ET - ENU
                        pass
            else:
                raise RuntimeError("No fission energy release coefficients.")
        elif isinstance(mf1mt458[comp], dict):
            # Get the tabulated form
            out[comp] = mf1mt458[comp]['EIFC']
            assert isinstance(out[comp], endf.Tabulated1D)
        else:
            raise TypeError(f"Uknown type {type(mf1mt458[comp])} for component {comp}")

    # We reomve ET so that we can re-construct it
    if 'ER' in out:
        out.pop('ER', None)
    
    # Reconstruct recoverable energy
    out['ER'] = lambda E, ET=out['ET'], ENU=out['ENU']: ET(E) - ENU(E)

    return out

def get_potential_scatter_xs(mat: endf.Material) -> float:
    """
    Computes the potential scattering cross section from the material.

    Parameters
    ----------
    mat : endf.Material
        Material for incident neutrons which must contain MF 2 MT 151.

    Returns
    -------
    float
        Potential scattering cross section in units of barns.

    Raises
    ------
    KeyError
        If the material does not contain MF 2 MT 151.
    RuntimeError
        If they scattering radius could not be determined.
    """
    if not (2, 151) in mat:
        raise KeyError("Material does not have MF 2 MT 151")
    mf2mt151 = mat[2,151]
    rr1 = mf2mt151['isotopes'][0]['ranges'][0]
    AP = 0.
    if 'AP' in rr1:
        AP = rr1['AP']
    
    # If there is an energy dependent radius, AP is still 0 !
    if AP == 0. and 'APE' in rr1:
        # Energy dependent scattering radius. Evaluate at 1keV
        assert isinstance(rr1['APE'], endf.Tabulated1D)
        AP = rr1['APE'](1.E3)
    
    if AP == 0. and rr1['LRF'] == 3:
        # Try for a spin group radius ?
        AP = rr1['sections'][0]['APL']
    elif AP == 0. and rr1['LRF'] == 7:
        # Get the first spin group, and grab the first
        # non-zero effective channel radius
        APE = rr1['spin_groups'][0]['channels']['APE']
        for val in APE:
            if val != 0.:
                AP = val
                break
    
    # If we get here, we couldn't figure it out based on the first resonance range.
    # Try to see if there is a second range, or a URR ?
    if AP == 0:
        # Check for a second resonance range !
        if len(mf2mt151['isotopes'][0]['ranges']) > 1:
            rr2 = mf2mt151['isotopes'][0]['ranges'][1]
            if 'AP' in rr2:
                AP = rr2['AP']
            elif 'APE' in rr2:
                assert isinstance(rr2['APE'], endf.Tabulated1D)
                AP = rr2['APE'](1.E3)
    
    # If we get here, we have lost all hope
    if AP == 0.:
        print(rr1)
        raise RuntimeError("Could not determine the scattering radius.")
                
    return 4.0 * np.pi * AP * AP

