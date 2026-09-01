// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title PostAnchor
 * @notice Immutable blockchain anchoring of verified social media posts.
 */
contract PostAnchor {
    struct Anchor {
        uint256 anchoredAt;
        address by;
        string postUrl;
    }

    mapping(bytes32 => Anchor) public anchors;

    event Anchored(bytes32 indexed contentHash, string postUrl, uint256 at, address by);

    function anchor(bytes32 contentHash, string calldata postUrl) external {
        require(anchors[contentHash].anchoredAt == 0, "already anchored");
        anchors[contentHash] = Anchor(block.timestamp, msg.sender, postUrl);
        emit Anchored(contentHash, postUrl, block.timestamp, msg.sender);
    }
}
