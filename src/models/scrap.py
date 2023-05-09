        
    """        

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
        assert V.shape[1] == 0, "The splitw_OB method should return the correct V component."
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
