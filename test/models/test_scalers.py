import pytest

from src.models.scalers import OBNScaler, OBPriceScaler
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
import numpy as np

class TestOBPriceScaler:    
    def test_obprice_scaler_init(self):
        price_scaler = StandardScaler()
        mask = np.array([True, False, True])
        ob_scaler = OBPriceScaler(price_scaler, mask)
        
        assert ob_scaler.price_scaler is price_scaler, "Price scaler should be initialized correctly."
        assert np.array_equal(ob_scaler.mask, mask), "Mask should be initialized correctly."
        assert ob_scaler.is_fitted_ is False, "is_fitted_ should be initialized as False."

    def test_obprice_scaler_fill_input(self):
        price_scaler = StandardScaler()
        mask = np.array([True, False, True, False])
        ob_scaler = OBPriceScaler(price_scaler, mask)
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        Xt = ob_scaler.fill_input(X)
        
        expected_Xt = np.array([[1, 0, 2, 0], [3, 0, 4, 0], [5, 0, 6, 0]])
        assert np.array_equal(Xt, expected_Xt), "The fill_input method should create a compatible input for price_scaler."
            
    def test_obprice_scaler_fit(self):
        price_scaler = StandardScaler()
        mask = np.array([True, False, True, False])
        ob_scaler = OBPriceScaler(price_scaler, mask)
        
        # Test check_is_fitted exception when price_scaler is not fitted
        X = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(NotFittedError):
            ob_scaler.fit(X)

        # Fit the price_scaler
        X_price_scaler = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        price_scaler.fit(X_price_scaler)
    
        # Fit the OBPriceScaler
        ob_scaler.fit(X)
    
        assert ob_scaler.is_fitted_ is True, "OBPriceScaler should be fitted after calling fit method."
        assert ob_scaler.n_features_in_ == 2, "OBPriceScaler should correctly compute the number of input features."
    
    def test_obprice_scaler_transform(self):
        price_scaler = StandardScaler()
        mask = np.array([True, False, True, False])
        ob_scaler = OBPriceScaler(price_scaler, mask)

        # Fit the price_scaler
        X_price_scaler = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        price_scaler.fit(X_price_scaler)

        # Fit the OBPriceScaler
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ob_scaler.fit(X)

        # Test the transform method
        transformed_X = ob_scaler.transform(X)
        expected_transformed_X = (X - price_scaler.mean_[mask]) / (np.sqrt(price_scaler.var_[mask]))
        
        assert np.allclose(transformed_X, expected_transformed_X), "The transform method should work correctly."
        
    def test_obprice_scaler_inverse_transform(self):
        price_scaler = StandardScaler()
        mask = np.array([True, False, True, False])
        ob_scaler = OBPriceScaler(price_scaler, mask)
        
        # Fit the price_scaler
        X_price_scaler = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        price_scaler.fit(X_price_scaler)
        
        # Fit the OBPriceScaler
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ob_scaler.fit(X)

        # Test the inverse_transform method
        transformed_X = ob_scaler.transform(X)
        inverse_transformed_X = ob_scaler.inverse_transform(transformed_X)
        assert np.allclose(inverse_transformed_X, X), "The inverse_transform method should work correctly."


class TestOBNScalerWithOB():
    def test_obn_scaler_init(self):
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 3
        indices = np.array([0, 2])
        N_INPUT = 7
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([4])
        po_indices = np.array([5])
        p_indices = np.array([6])
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter)

        assert obn_scaler.scaler_ == scaler_, "The provided scaler should be stored correctly."
        assert obn_scaler.volume_scaler_ == volume_scaler_, "The provided volume_scaler should be stored correctly."
        assert obn_scaler.OBs == OBs, "The provided OBs should be stored correctly."
        assert np.array_equal(obn_scaler.indices, indices), "The provided indices should be stored correctly."
        assert obn_scaler.N_INPUT == N_INPUT, "The provided N_INPUT should be stored correctly."
        assert obn_scaler.N_DATA == N_DATA, "The provided N_DATA should be stored correctly."
        assert np.array_equal(obn_scaler.x_indices, x_indices), "The provided x_indices should be stored correctly."
        assert np.array_equal(obn_scaler.v_indices, v_indices), "The provided v_indices should be stored correctly."
        assert np.array_equal(obn_scaler.po_indices, po_indices), "The provided po_indices should be stored correctly."
        assert np.array_equal(obn_scaler.p_indices, p_indices), "The provided p_indices should be stored correctly."
        assert obn_scaler.spliter == spliter, "The provided spliter should be stored correctly."

    def test_obn_scaler_split_OB(self):
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 3
        nh = 2
        indices = np.array([0, 2])
        N_INPUT = 22
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([4, 5, 6, 7, 8, 9])
        po_indices = np.array([10, 11, 12, 13, 14, 15])
        p_indices = np.array([16, 17, 18, 19, 20, 21])
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter, nh=nh)

        X_input = np.array([
            [1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322]
        ])

        X, V, Po, P = obn_scaler.split_OB(X_input)
        
        expected_X = np.array([
            [1, 2, 3, 4],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
        ])
        
        expected_V = np.array([
            [5, 6, 7, 8, 9, 10],
            [25, 26, 27, 28, 29, 210],
            [35, 36, 37, 38, 39, 310]
        ])

        expected_Po = np.array([
            [11, 14],
            [12, 15],            
            [13, 16],
            [211, 214],
            [212, 215],            
            [213, 216],
            [311, 314],
            [312, 315],            
            [313, 316],      
        ])

        expected_P = np.array([
            [17, 20], 
            [18, 21],           
            [19, 22],
            [217, 220], 
            [218, 221],           
            [219, 222],
            [317, 320], 
            [318, 321],           
            [319, 322],            
        ])

        assert np.array_equal(X, expected_X), "The split_OB method should return the correct X component."
        assert np.array_equal(V, expected_V), "The split_OB method should return the correct V component."
        assert np.array_equal(Po, expected_Po), "The split_OB method should return the correct Po component."
        assert np.array_equal(P, expected_P), "The split_OB method should return the correct P component."

    def test_obn_scaler_reconstruct_OB(self):
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 3
        nh = 2
        indices = np.array([0, 2])
        N_INPUT = 22
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([4, 5, 6, 7, 8, 9])
        po_indices = np.array([10, 11, 12, 13, 14, 15])
        p_indices = np.array([16, 17, 18, 19, 20, 21])
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter, nh=nh)

        X_input = np.array([
            [1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322]
        ])

        X, V, Po, P = obn_scaler.split_OB(X_input)        
        Xt = obn_scaler.reconstruct_OB(X, V, Po, P)
        
        assert np.array_equal(Xt, X_input)

    def test_obn_scaler_fit(self):
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 3
        nh = 2
        indices = np.array([0, 2])
        N_INPUT = 22
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([4, 5, 6, 7, 8, 9])
        po_indices = np.array([10, 11, 12, 13, 14, 15])
        p_indices = np.array([16, 17, 18, 19, 20, 21])
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter, nh=nh)

        X_input = np.array([
            [1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322]
        ])        

        obn_scaler.fit(X_input)

        # Check the inside scaler : regular scaler
        assert obn_scaler.scaler.is_fitted_, "The fit method should set is_fitted_ attribute to True."
        assert hasattr(obn_scaler.scaler, 'n_features_in_')        

        # Check the inside scaler : volume scaler
        assert obn_scaler.volume_scaler.is_fitted_, "The fit method should set is_fitted_ attribute to True."
        assert hasattr(obn_scaler.volume_scaler, 'n_features_in_')        

        # Check the inside scaler : price scaler
        assert obn_scaler.price_scaler.is_fitted_, "The fit method should set is_fitted_ attribute to True."
        assert hasattr(obn_scaler.price_scaler, 'n_features_in_')        

        # Main scaler
        assert obn_scaler.is_fitted_, "The fit method should set is_fitted_ attribute to True."
        assert hasattr(obn_scaler, 'n_features_in_')

    def test_obn_scaler_transform(self):
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 3
        nh = 2
        indices = np.array([0, 2])
        N_INPUT = 22
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([4, 5, 6, 7, 8, 9])
        po_indices = np.array([10, 11, 12, 13, 14, 15])
        p_indices = np.array([16, 17, 18, 19, 20, 21])
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter, nh=nh)

        X_input = np.array([
            [1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322]
        ])        

        obn_scaler.fit(X_input)
        X_transformed = obn_scaler.transform(X_input)

        assert np.allclose(
            X_transformed[:, obn_scaler.x_indices],
            obn_scaler.scaler.transform(X_input[:,obn_scaler.x_indices]),atol=1e-8)
        assert np.allclose(
            X_transformed[:, obn_scaler.v_indices],
            obn_scaler.volume_scaler.transform(X_input[:,obn_scaler.v_indices]),
            atol=1e-8)

        X, V, Po, P = obn_scaler.split_OB(X_input)
        Pot = obn_scaler.price_scaler.transform(Po)
        Pt = obn_scaler.price_scaler.transform(P)

        assert obn_scaler.price_scaler.price_scaler is obn_scaler.scaler

        expected_Pot = (Po - obn_scaler.scaler.scaler.mean_[obn_scaler.indices]) / np.sqrt(obn_scaler.scaler.scaler.var_[obn_scaler.indices])
        expected_Pt = (P - obn_scaler.scaler.scaler.mean_[obn_scaler.indices]) / np.sqrt(obn_scaler.scaler.scaler.var_[obn_scaler.indices])        
        assert np.allclose(Pot, expected_Pot, atol=1e-8), "The transform method should produce the expected transformed data."
        assert np.allclose(Pt, expected_Pt, atol=1e-8), "The transform method should produce the expected transformed data."                 

 
    def test_obn_scaler_inverse_transform(self):
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 3
        nh = 2
        indices = np.array([0, 2])
        N_INPUT = 22
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([4, 5, 6, 7, 8, 9])
        po_indices = np.array([10, 11, 12, 13, 14, 15])
        p_indices = np.array([16, 17, 18, 19, 20, 21])
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter, nh=nh)

        X_input = np.array([
            [1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322]
        ])        

        obn_scaler.fit(X_input)
        X_transformed = obn_scaler.transform(X_input)
        X_reconstructed = obn_scaler.inverse_transform(X_transformed)
        assert np.allclose(X_input, X_reconstructed, atol=1e-8), "The inverse transform method of the transformed data should yield the original data."
        

class TestOBNScalerWithoutOB():
    def test_obn_scaler_init(self):
        nh = 24        
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 20
        indices = np.array([0, 2])
        N_INPUT = 4
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([], dtype=bool)
        po_indices = np.array([], dtype=bool)
        p_indices = np.array([], dtype=bool)
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter)

        assert obn_scaler.scaler_ == scaler_, "The provided scaler should be stored correctly."
        assert obn_scaler.volume_scaler_ == volume_scaler_, "The provided volume_scaler should be stored correctly."
        assert obn_scaler.OBs == OBs, "The provided OBs should be stored correctly."
        assert np.array_equal(obn_scaler.indices, indices), "The provided indices should be stored correctly."
        assert obn_scaler.N_INPUT == N_INPUT, "The provided N_INPUT should be stored correctly."
        assert obn_scaler.N_DATA == N_DATA, "The provided N_DATA should be stored correctly."
        assert np.array_equal(obn_scaler.x_indices, x_indices), "The provided x_indices should be stored correctly."
        assert np.array_equal(obn_scaler.v_indices, v_indices), "The provided v_indices should be stored correctly."
        assert np.array_equal(obn_scaler.po_indices, po_indices), "The provided po_indices should be stored correctly."
        assert np.array_equal(obn_scaler.p_indices, p_indices), "The provided p_indices should be stored correctly."
        assert obn_scaler.spliter == spliter, "The provided spliter should be stored correctly."

    def test_obn_scaler_split_OB(self):
        nh = 24        
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 20
        indices = np.array([0, 2])
        N_INPUT = 4
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([], dtype=bool)
        po_indices = np.array([], dtype=bool)
        p_indices = np.array([], dtype=bool)
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter, nh=nh)

        X_input = np.array([
            [1,2,3,4],
            [21, 22, 23, 24],
            [31, 32, 33, 34]
        ])

        X, V, Po, P = obn_scaler.split_OB(X_input)
        
        expected_X = np.array([
            [1, 2, 3, 4],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
        ])
        
        assert np.array_equal(X, expected_X), "The split_OB method should return the correct X component."
        assert V.shape[1] == 0, "The split_OB method should return the correct V component."
        assert Po.shape[1] == 0, "The split_OB method should return the correct Po component."
        assert P.shape[1] == 0, "The split_OB method should return the correct P component."

    def test_obn_scaler_fit(self):
        nh = 24        
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 20
        indices = np.array([0, 2])
        N_INPUT = 4
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([], dtype=bool)
        po_indices = np.array([], dtype=bool)
        p_indices = np.array([], dtype=bool)
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter, nh=nh)

        X_input = np.array([
            [1,2,3,4],
            [21, 22, 23, 24],
            [31, 32, 33, 34]
        ])        

        obn_scaler.fit(X_input)

        # Check the inside scaler : regular scaler
        assert obn_scaler.scaler.is_fitted_, "The fit method should set is_fitted_ attribute to True."
        assert hasattr(obn_scaler.scaler, 'n_features_in_')        

        # Check the inside scaler : volume scaler
        assert not obn_scaler.volume_scaler.is_fitted_, "The fit method should do nothing for the volume scaler."
        # Check the inside scaler : price scaler
        assert not obn_scaler.price_scaler.is_fitted_, "The fit method should do nothing for the volume scaler"

        # Main scaler
        assert obn_scaler.is_fitted_, "The fit method should set is_fitted_ attribute to True."
        assert hasattr(obn_scaler, 'n_features_in_')

    def test_obn_scaler_transform(self):
        nh = 24        
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 20
        indices = np.array([0, 2])
        N_INPUT = 4
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([], dtype=bool)
        po_indices = np.array([], dtype=bool)
        p_indices = np.array([], dtype=bool)
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter, nh=nh)

        X_input = np.array([
            [1,2,3,4],
            [21, 22, 23, 24],
            [31, 32, 33, 34]
        ])        

        obn_scaler.fit(X_input)
        X_transformed = obn_scaler.transform(X_input)

        assert np.allclose(
            X_transformed[:, obn_scaler.x_indices],
            obn_scaler.scaler.transform(X_input[:,obn_scaler.x_indices]),atol=1e-8)
        
    def test_obn_scaler_inverse_transform(self):
        nh = 24        
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 20
        indices = np.array([0, 2])
        N_INPUT = 4
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([], dtype=bool)
        po_indices = np.array([], dtype=bool)
        p_indices = np.array([], dtype=bool)
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter, nh=nh)

        X_input = np.array([
            [1,2,3,4],
            [21, 22, 23, 24],
            [31, 32, 33, 34]
        ])     

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
    def test_obn_scaler_inverse_transform(self):
        scaler_ = "Standard"
        volume_scaler_ = "Standard"
        OBs = 3
        nh = 2
        indices = np.array([0, 2])
        N_INPUT = 22
        N_DATA = 4
        x_indices = np.array([0, 1, 2, 3])
        v_indices = np.array([4, 5, 6, 7, 8, 9])
        po_indices = np.array([10, 11, 12, 13, 14, 15])
        p_indices = np.array([16, 17, 18, 19, 20, 21])
        spliter = None

        obn_scaler = OBNScaler(
            scaler_, volume_scaler_, OBs, indices, N_INPUT,
            N_DATA, x_indices, v_indices, po_indices, p_indices, spliter, nh=nh)

        X_input_fit = np.array([
            [1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
             22],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212, 213, 214, 215, 216,
             217, 218, 219, 220, 221, 222],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311, 312, 313, 314, 315, 316,
             317, 318, 319, 320, 321, 322]
        ])        

        obn_scaler.fit(X_input_fit)
        X_transformed = obn_scaler.transform(X_input_fit)[:, np.concatenate(
            (v_indices, po_indices, p_indices))]
        X_reconstructed = obn_scaler.inverse_transform(X_transformed)

        X_expected = np.array([
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
             22],
            [25, 26, 27, 28, 29, 210, 211, 212, 213, 214, 215, 216,
             217, 218, 219, 220, 221, 222],
            [35, 36, 37, 38, 39, 310, 311, 312, 313, 314, 315, 316,
             317, 318, 319, 320, 321, 322]
        ])
        print(X_transformed.shape, X_reconstructed.shape, X_expected.shape)
        assert np.allclose(X_expected, X_reconstructed, atol=1e-8), "The inverse transform method of the transformed data should yield the original data."
        
