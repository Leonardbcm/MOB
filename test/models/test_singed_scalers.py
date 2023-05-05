import pytest

from src.models.scalers import SignOBNScaler
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
import numpy as np


class TestOBNScalerWithOB():
    def get_obn(self):
        scaler_ = "Standard"
        OBs = 2
        N_DATA = 3
        nh = 2        
        N_INPUT = N_DATA + OBs * 3 * nh
        x_indices = np.array([0, 1, 2])
        v_indices = np.array([3, 4, 9, 10])
        po_indices = np.array([5, 6, 11, 12])
        p_indices = np.array([7, 8, 13, 14])
        spliter = None

        obn_scaler = SignOBNScaler(
            scaler_, OBs, N_INPUT, N_DATA, x_indices, v_indices,
            po_indices, p_indices, spliter=None, nh=nh)
        return obn_scaler

    def get_xinput(self):
        X_input = np.array([
            [1,2,3, 4, -5, 6, 7, 8, -9, 10, -11, 12, 13, 14, -15],
            [21,22, 23, 24, -25, 26, 27, 28, -29, 210, -211, 212, 213, 214, -215],
            [31, 32, 33, 34, -35, 36, 37, 38, -39, 310, -311, 312, 313, 314, -315]
        ])
        return X_input
        
    def test_obn_scaler_init(self):
        obn_scaler = self.get_obn()
        
        assert np.array_equal(obn_scaler.mask_unit, np.array([1, -1], dtype=int))
        assert np.array_equal(obn_scaler.mask, np.array([1, -1, 1, -1], dtype=int))
        assert np.array_equal(
            obn_scaler.OB_indices,
            np.array([3,4,5,6,7,8,9,10,11,12, 13, 14], dtype=int))

    def test_obn_scaler_split_OB(self):
        obn_scaler = self.get_obn()
        X_input  = self.get_xinput()        
        X, OB = obn_scaler.split_OB(X_input)

        V = X_input[:, obn_scaler.v_indices]
        Po = X_input[:, obn_scaler.po_indices]
        P = X_input[:, obn_scaler.p_indices]        
        
        expected_X = np.array([
            [1, 2, 3],
            [21, 22, 23],
            [31, 32, 33],
        ])
        
        expected_V = np.array([
            [4, -5, 10, -11],
            [24, -25, 210, -211],
            [34, -35, 310, -311],            
        ])

        expected_Po = np.array([
            [6, 7, 12, 13],
            [26, 27, 212, 213],
            [36, 37, 312, 313],            
        ])

        expected_P = np.array([
            [8, -9, 14, -15],
            [28, -29, 214, -215],
            [38, -39, 314, -315],             
        ])

        expected_OB = np.array([
            [4, -5, 6, 7, 8, -9, 10, -11, 12, 13, 14, -15],
            [24, -25, 26, 27, 28, -29, 210, -211, 212, 213, 214, -215],
            [34, -35, 36, 37, 38, -39, 310, -311, 312, 313, 314, -315]
        ])
        assert np.array_equal(X, expected_X), "The split_OB method should return the correct X component."
        assert np.array_equal(V, expected_V), "The split_OB method should return the correct V component."
        assert np.array_equal(Po, expected_Po), "The split_OB method should return the correct Po component."
        assert np.array_equal(P, expected_P), "The split_OB method should return the correct P component."
        assert np.array_equal(OB, expected_OB), "The split_OB method should return the correct OB component."
        
    def test_obn_scaler_reconstruct_OB(self):
        obn_scaler = self.get_obn()
        X_input  = self.get_xinput()        
        X, OB = obn_scaler.split_OB(X_input)        
        Xt = obn_scaler.reconstruct_OB(X, OB)
        
        assert np.array_equal(Xt, X_input)

    def test_obn_scaler_fit(self):
        obn_scaler = self.get_obn()
        X_input  = self.get_xinput()        

        obn_scaler.fit(X_input)

        # Check the inside scaler : regular scaler
        assert obn_scaler.scaler.is_fitted_, "The fit method should set is_fitted_ attribute to True."
        assert hasattr(obn_scaler.scaler, 'n_features_in_')        

        # Check the inside scaler : volume scaler
        assert hasattr(obn_scaler.OB_scaler, 'n_features_in_')

    def test_obn_scaler_transform(self):
        obn_scaler = self.get_obn()
        X_input  = self.get_xinput()        

        obn_scaler.fit(X_input)
        X_transformed = obn_scaler.transform(X_input)
        
        # Expected signs
        v_signs = np.array([1, -1, 1, -1], dtype=int)
        p_signs = np.array([1, -1, 1, -1], dtype=int)

        assert np.array_equal(v_signs,np.sign(X_transformed[1,obn_scaler.v_indices]))
        assert np.array_equal(p_signs,np.sign(X_transformed[1,obn_scaler.p_indices]))        

    def test_obn_scaler_inverse_transform(self):
        obn_scaler = self.get_obn()
        X_input  = self.get_xinput()        

        obn_scaler.fit(X_input)
        X_transformed = obn_scaler.transform(X_input)
        X_reconstructed = obn_scaler.inverse_transform(X_transformed)
        assert np.allclose(X_input, X_reconstructed, atol=1e-8), "The inverse transform method of the transformed data should yield the original data."
        

class TestOBNScalerWithoutOB():
    def get_obn(self):
        scaler_ = "Standard"
        OBs = 2
        N_DATA = 3
        nh = 2        
        N_INPUT = N_DATA
        x_indices = np.array([0, 1, 2])
        v_indices = np.array([], dtype=int)
        po_indices = np.array([], dtype=int)
        p_indices = np.array([], dtype=int)
        spliter = None

        obn_scaler = SignOBNScaler(
            scaler_, OBs, N_INPUT, N_DATA, x_indices, v_indices,
            po_indices, p_indices, spliter=None, nh=nh)
        return obn_scaler

    def get_xinput(self):
        X_input = np.array([
            [1,  2,  3],
            [21, 22, 23],
            [31, 32, 33]
        ])
        return X_input
        
    def test_obn_scaler_init(self):
        obn_scaler = self.get_obn()
        
        assert np.array_equal(obn_scaler.mask_unit, np.array([1, -1], dtype=int))
        assert np.array_equal(obn_scaler.mask, np.array([1, -1, 1, -1], dtype=int))
        assert np.array_equal(
            obn_scaler.OB_indices,
            np.array([], dtype=int))

    def test_obn_scaler_split_OB(self):
        obn_scaler = self.get_obn()
        X_input  = self.get_xinput()        
        X, OB = obn_scaler.split_OB(X_input)

        expected_X = np.array([
            [1, 2, 3],
            [21, 22, 23],
            [31, 32, 33],
        ])
        
        assert np.array_equal(X, expected_X), "The split_OB method should return the correct X component."
        assert OB.shape[1] == 0, "The split_OB method should return the correct OB component."
        
    def test_obn_scaler_reconstruct_OB(self):
        obn_scaler = self.get_obn()
        X_input  = self.get_xinput()        
        X, OB = obn_scaler.split_OB(X_input)        
        Xt = obn_scaler.reconstruct_OB(X, OB)
        
        assert np.array_equal(Xt, X_input)

    def test_obn_scaler_fit(self):
        obn_scaler = self.get_obn()
        X_input  = self.get_xinput()        

        obn_scaler.fit(X_input)

        # Check the inside scaler : regular scaler
        assert obn_scaler.scaler.is_fitted_, "The fit method should set is_fitted_ attribute to True."
        assert hasattr(obn_scaler.scaler, 'n_features_in_')        
        
    def test_obn_scaler_inverse_transform(self):
        obn_scaler = self.get_obn()
        X_input  = self.get_xinput()        

        obn_scaler.fit(X_input)
        X_transformed = obn_scaler.transform(X_input)
        X_reconstructed = obn_scaler.inverse_transform(X_transformed)
        assert np.allclose(X_input, X_reconstructed, atol=1e-8), "The inverse transform method of the transformed data should yield the original data."
        

class TestOBNScalerOnlyOB():
    """
    Case of predict order books: prices are in the input data and are used for
    fitting, but are not in the data to inverse_transform.

    We only test the inverse transform method because other behave similarly
    """
    def get_obn(self):
        scaler_ = "Standard"
        OBs = 2
        N_DATA = 3
        nh = 2        
        N_INPUT = N_DATA + OBs * 3 * nh
        x_indices = np.array([0, 1, 2])
        v_indices = np.array([3, 4, 9, 10])
        po_indices = np.array([5, 6, 11, 12])
        p_indices = np.array([7, 8, 13, 14])
        spliter = None

        obn_scaler = SignOBNScaler(
            scaler_, OBs, N_INPUT, N_DATA, x_indices, v_indices,
            po_indices, p_indices, spliter=None, nh=nh)
        return obn_scaler

    def get_xinput(self):
        X_input = np.array([
            [1,2,3, 4, -5, 6, 7, 8, -9, 10, -11, 12, 13, 14, -15],
            [21,22, 23, 24, -25, 26, 27, 28, -29, 210, -211, 212, 213, 214, -215],
            [31, 32, 33, 34, -35, 36, 37, 38, -39, 310, -311, 312, 313, 314, -315]
        ])
        return X_input
    
    def test_obn_scaler_inverse_transform(self):
        obn_scaler = self.get_obn()
        X_input = self.get_xinput()
        _, OB_input = obn_scaler.split_OB(X_input)        
        
        obn_scaler.fit(X_input)
        X_transformed = obn_scaler.transform(X_input)
        _, OB_transformed = obn_scaler.split_OB(X_transformed)

        X_reconstructed = obn_scaler.inverse_transform(OB_transformed)
        print(X_input.shape, X_transformed.shape, X_reconstructed.shape)
        assert np.allclose(OB_input, X_reconstructed, atol=1e-8), "The inverse transform method of the transformed data should yield the original data."
        
