

namespace Testing
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void TestLocalModel()
        {
            ImageRecognition.Accessor.Predict_LocalModel("path1", "path2");
        }
    }
}