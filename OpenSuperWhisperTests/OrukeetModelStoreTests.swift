import XCTest
@testable import OpenSuperWhisper

final class OrukeetModelStoreTests: XCTestCase {
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func manifest(filename: String = "orukeet-r3-coreml-baseline.zip",
                          bytes: Int = OrukeetModelStore.archiveBytes,
                          sha256: String = OrukeetModelStore.archiveSHA256) throws -> Data {
        try JSONSerialization.data(withJSONObject: ["archives": ["baseline": [
            "filename": filename, "bytes": bytes, "sha256": sha256,
        ]]])
    }

    func testManifestRejectsChangedArtifactMetadata() throws {
        XCTAssertNoThrow(try OrukeetModelStore.validateManifest(manifest()))
        XCTAssertThrowsError(try OrukeetModelStore.validateManifest(manifest(filename: "other.zip")))
        XCTAssertThrowsError(try OrukeetModelStore.validateManifest(manifest(bytes: 1)))
        XCTAssertThrowsError(try OrukeetModelStore.validateManifest(manifest(sha256: "incorrect")))
        XCTAssertThrowsError(try OrukeetModelStore.validateManifest(Data("{}".utf8)))
    }

    func testIncompleteCacheIsNotInstalled() throws {
        let directory = try temporaryDirectory()
        try OrukeetModelStore.revision.write(to: directory.appendingPathComponent(".revision"),
                                            atomically: true, encoding: .utf8)
        XCTAssertFalse(OrukeetModelStore.installed(at: directory))
        XCTAssertThrowsError(try OrukeetModelStore.load(from: directory))
    }

    func testCorruptArchivePreservesExistingCache() throws {
        let directory = try temporaryDirectory()
        let destination = directory.appendingPathComponent("installed")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let marker = destination.appendingPathComponent("previous")
        try Data("valid previous install".utf8).write(to: marker)
        let archive = directory.appendingPathComponent("corrupt.zip")
        try Data("corrupt".utf8).write(to: archive)
        XCTAssertThrowsError(try OrukeetModelStore.installArchive(at: archive, to: destination))
        XCTAssertEqual(try Data(contentsOf: marker), Data("valid previous install".utf8))
    }

    func testFailedCommitRestoresExistingCache() throws {
        let directory = try temporaryDirectory()
        let destination = directory.appendingPathComponent("installed")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let marker = destination.appendingPathComponent("previous")
        try Data("previous".utf8).write(to: marker)
        XCTAssertThrowsError(try OrukeetModelStore.commitInstallation(
            from: directory.appendingPathComponent("missing"), to: destination))
        XCTAssertEqual(try Data(contentsOf: marker), Data("previous".utf8))
    }

    func testOrukeetHasItsOwnSelectionAndLanguages() throws {
        let model = try XCTUnwrap(SettingsFluidAudioModels.availableModels.first { $0.version == "orukeet" })
        XCTAssertEqual(model.name, "Orukeet (preview)")
        XCTAssertFalse(model.isDownloaded)
        let languages = LanguageUtil.supportedLanguages(engine: "fluidaudio", fluidAudioModelVersion: model.version)
        XCTAssertEqual(Set(languages).count, 25)
        XCTAssertTrue(languages.contains("mt"))
        XCTAssertTrue(languages.contains("el"))
        XCTAssertFalse(languages.contains("zh"))
    }

    // Opt in with the immutable archive downloaded from the model card. Default CI uses no weights.
    func testPinnedArchiveInstallsAndLoadsOffline() throws {
        guard let path = ProcessInfo.processInfo.environment["ORUKEET_COREML_ARCHIVE"] else {
            throw XCTSkip("Set ORUKEET_COREML_ARCHIVE to run the real Core ML installation test")
        }
        let destination = try temporaryDirectory().appendingPathComponent("installed")
        try OrukeetModelStore.installArchive(at: URL(fileURLWithPath: path), to: destination)
        XCTAssertTrue(OrukeetModelStore.installed(at: destination))
        XCTAssertNoThrow(try OrukeetModelStore.load(from: destination))
        try FileManager.default.removeItem(at: destination.appendingPathComponent("Encoder.mlmodelc"))
        XCTAssertFalse(OrukeetModelStore.installed(at: destination))
        XCTAssertThrowsError(try OrukeetModelStore.load(from: destination))
    }
}
